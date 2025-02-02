from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import logfire
from devtools import debug
from httpx import AsyncClient
from dotenv import load_dotenv

from supabase import Client
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
llm = os.getenv('LLM_MODEL', 'gpt-4o')

client = AsyncOpenAI(
    # base_url = 'http://localhost:11434/v1',
    # api_key='ollama'
)

model = OpenAIModel(llm) if llm.lower().startswith("gpt") else OpenAIModel(llm, openai_client=client)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


@dataclass
class WebResearcherDeps:
    client: AsyncClient
    supabase: Client
    session_id: str
    brave_api_key: str | None


web_search_agent = Agent(
    model,
    system_prompt=f'You are an expert at researching the web to answer user questions. The current date is: {datetime.now().strftime("%Y-%m-%d")}',
    deps_type=WebResearcherDeps,
    retries=2
)


@web_search_agent.tool
async def search_web(
    ctx: RunContext[WebResearcherDeps], web_query: str
) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."

    headers = {
        'X-Subscription-Token': ctx.deps.brave_api_key,
        'Accept': 'application/json',
    }
    
    with logfire.span('calling Brave search API', query=web_query) as span:
        ctx.deps.supabase.table("messages").insert({
            "session_id": ctx.deps.session_id,
            "message": {
                "type": "ai",
                "content": f"Researching web with query: {web_query}"
            }
        }).execute()

        r = await ctx.deps.client.get(
            'https://api.search.brave.com/res/v1/web/search',
            params={
                'q': web_query,
                'count': 5,
                'text_decorations': True,
                'search_lang': 'en'
            },
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    results = []
    
    # Add web results in a nice formatted way
    web_results = data.get('web', {}).get('results', [])
    for item in web_results[:3]:
        title = item.get('title', '')
        description = item.get('description', '')
        url = item.get('url', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."


async def main():
    async with AsyncClient() as client:
        brave_api_key = os.getenv('BRAVE_API_KEY', None)
        deps = WebResearcherDeps(client=client, brave_api_key=brave_api_key)

        result = await web_search_agent.run(
            'Give me some articles talking about the new release of React 19.', deps=deps
        )
        
        debug(result)
        print('Response:', result.data)


if __name__ == '__main__':
    asyncio.run(main())