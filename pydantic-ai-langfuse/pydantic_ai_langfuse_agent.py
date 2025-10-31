from __future__ import annotations
from contextlib import AsyncExitStack
from typing import Any, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import os

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent, RunContext

from configure_langfuse import configure_langfuse

load_dotenv()

# Configure Langfuse for agent observability (provide a no-op if not configured)
try:
    tracer = configure_langfuse()
except Exception:
    class _NoOpSpan:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def set_attribute(self, *a, **k):
            return None

    class _NoOpTracer:
        def start_as_current_span(self, name):
            return _NoOpSpan()

    tracer = _NoOpTracer()

# Read configuration for running MCP servers vs local-only
USE_MCP_SERVERS = os.getenv("USE_MCP_SERVERS", "true").lower() in ("1", "true", "yes")


# Helper to safely build MCP server instances only when enabled
def build_mcp_server(command: str, args: list[str], env: dict[str, str] | None = None) -> MCPServerStdio | None:
    if not USE_MCP_SERVERS:
        return None
    kwargs = {"env": env} if env is not None else {}
    return MCPServerStdio(command, args, **kwargs)

# ========== Helper function to get model configuration ==========
def get_model():
    llm = os.getenv('MODEL_CHOICE', 'gpt-4.1-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

# ========== Set up MCP servers for each service ==========

# Brave Search MCP server
brave_server = build_mcp_server('npx', ['-y', '@modelcontextprotocol/server-brave-search'],
                                {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")})

# Airtable MCP server
airtable_server = build_mcp_server('npx', ['-y', 'airtable-mcp-server'],
                                  {"AIRTABLE_API_KEY": os.getenv("AIRTABLE_API_KEY")})

# Filesystem MCP server
filesystem_server = build_mcp_server('npx', ['-y', '@modelcontextprotocol/server-filesystem', os.getenv("LOCAL_FILE_DIR")])

# GitHub MCP server
github_server = build_mcp_server('npx', ['-y', '@modelcontextprotocol/server-github'],
                                {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")})

# Slack MCP server
slack_server = build_mcp_server('npx', ['-y', '@modelcontextprotocol/server-slack'],
                               {
                                   "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
                                   "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID")
                               })

# Firecrawl MCP server
firecrawl_server = build_mcp_server('npx', ['-y', 'firecrawl-mcp'],
                                   {"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")})

# ========== Create subagents with their MCP servers ==========

# Brave search agent
brave_agent = Agent(
    get_model(),
    system_prompt="You are a web search specialist using Brave Search. Find relevant information on the web.",
    mcp_servers=[s for s in (brave_server,) if s is not None],
    instrument=True
)

# Airtable agent
airtable_agent = Agent(
    get_model(),
    system_prompt="You are an Airtable specialist. Help users interact with Airtable databases.",
    mcp_servers=[s for s in (airtable_server,) if s is not None],
    instrument=True
)

# Filesystem agent
filesystem_agent = Agent(
    get_model(),
    system_prompt="You are a filesystem specialist. Help users manage their files and directories.",
    mcp_servers=[s for s in (filesystem_server,) if s is not None],
    instrument=True
)

# GitHub agent
github_agent = Agent(
    get_model(),
    system_prompt="You are a GitHub specialist. Help users interact with GitHub repositories and features.",
    mcp_servers=[s for s in (github_server,) if s is not None],
    instrument=True
)

# Slack agent
slack_agent = Agent(
    get_model(),
    system_prompt="You are a Slack specialist. Help users interact with Slack workspaces and channels.",
    mcp_servers=[s for s in (slack_server,) if s is not None],
    instrument=True
)

# Firecrawl agent
firecrawl_agent = Agent(
    get_model(),
    system_prompt="You are a web crawling specialist. Help users extract data from websites.",
    mcp_servers=[s for s in (firecrawl_server,) if s is not None],
    instrument=True
)

# ========== Create the primary orchestration agent ==========
primary_agent = Agent(
    get_model(),
    system_prompt="""You are a primary orchestration agent that can call upon specialized subagents 
    to perform various tasks. Each subagent is an expert in interacting with a specific third-party service.
    Analyze the user request and delegate the work to the appropriate subagent.""",
    instrument=True
)

# ========== Define tools for the primary agent to call subagents ==========
@primary_agent.tool_plain
async def use_brave_search_agent(query: str) -> dict[str, str]:
    """
    Search the web using Brave Search through the Brave subagent.
    Use this tool when the user needs to find information on the internet or research a topic.

    Args:
        ctx: The run context.
        query: The search query or instruction for the Brave search agent.

    Returns:
        The search results or response from the Brave agent.
    """
    print(f"Calling Brave agent with query: {query} - pydantic_ai_langfuse_agent.py:159")
    if not getattr(brave_agent, "mcp_servers", []):
        return {"error": "Brave MCP server is disabled. Set USE_MCP_SERVERS=true to enable."}
    result = await brave_agent.run(query)
    return {"result": result.data}

@primary_agent.tool_plain
async def use_airtable_agent(query: str) -> dict[str, str]:
    """
    Access and manipulate Airtable data through the Airtable subagent.
    Use this tool when the user needs to fetch, modify, or analyze data in Airtable.

    Args:
        ctx: The run context.
        query: The instruction for the Airtable agent.

    Returns:
        The response from the Airtable agent.
    """
    print(f"Calling Airtable agent with query: {query} - pydantic_ai_langfuse_agent.py:176")
    if not getattr(airtable_agent, "mcp_servers", []):
        return {"error": "Airtable MCP server is disabled. Set USE_MCP_SERVERS=true to enable."}
    result = await airtable_agent.run(query)
    return {"result": result.data}

@primary_agent.tool_plain
async def use_filesystem_agent(query: str) -> dict[str, str]:
    """
    Interact with the file system through the filesystem subagent.
    Use this tool when the user needs to read, write, list, or modify files.

    Args:
        ctx: The run context.
        query: The instruction for the filesystem agent.

    Returns:
        The response from the filesystem agent.
    """
    print(f"Calling Filesystem agent with query: {query} - pydantic_ai_langfuse_agent.py:193")
    if not getattr(filesystem_agent, "mcp_servers", []):
        return {"error": "Filesystem MCP server is disabled. Set USE_MCP_SERVERS=true to enable."}
    result = await filesystem_agent.run(query)
    return {"result": result.data}

@primary_agent.tool_plain
async def use_github_agent(query: str) -> dict[str, str]:
    """
    Interact with GitHub through the GitHub subagent.
    Use this tool when the user needs to access repositories, issues, PRs, or other GitHub resources.

    Args:
        ctx: The run context.
        query: The instruction for the GitHub agent.

    Returns:
        The response from the GitHub agent.
    """
    print(f"Calling GitHub agent with query: {query} - pydantic_ai_langfuse_agent.py:210")
    if not getattr(github_agent, "mcp_servers", []):
        return {"error": "GitHub MCP server is disabled. Set USE_MCP_SERVERS=true to enable."}
    result = await github_agent.run(query)
    return {"result": result.data}

@primary_agent.tool_plain
async def use_slack_agent(query: str) -> dict[str, str]:
    """
    Interact with Slack through the Slack subagent.
    Use this tool when the user needs to send messages, access channels, or retrieve Slack information.

    Args:
        ctx: The run context.
        query: The instruction for the Slack agent.

    Returns:
        The response from the Slack agent.
    """
    print(f"Calling Slack agent with query: {query} - pydantic_ai_langfuse_agent.py:227")
    if not getattr(slack_agent, "mcp_servers", []):
        return {"error": "Slack MCP server is disabled. Set USE_MCP_SERVERS=true to enable."}
    result = await slack_agent.run(query)
    return {"result": result.data}

@primary_agent.tool_plain
async def use_firecrawl_agent(query: str) -> dict[str, str]:
    """
    Crawl and analyze websites using the Firecrawl subagent.
    Use this tool when the user needs to extract data from websites or perform web scraping.

    Args:
        ctx: The run context.
        query: The instruction for the Firecrawl agent.

    Returns:
        The response from the Firecrawl agent.
    """
    print(f"Calling Firecrawl agent with query: {query} - pydantic_ai_langfuse_agent.py:244")
    if not getattr(firecrawl_agent, "mcp_servers", []):
        return {"error": "Firecrawl MCP server is disabled. Set USE_MCP_SERVERS=true to enable."}
    result = await firecrawl_agent.run(query)
    return {"result": result.data}

# ========== Main execution function ==========

async def main():
    """Run the primary agent with a given query."""
    print("MCP Agent Army  Multiagent system using Model Context Protocol - pydantic_ai_langfuse_agent.py:252")
    print("Enter 'exit' to quit the program. - pydantic_ai_langfuse_agent.py:253")

    # Use AsyncExitStack to manage all MCP servers in one context
    async with AsyncExitStack() as stack:
        # Start MCP servers only when enabled and configured
        if USE_MCP_SERVERS:
            print("Starting MCP servers... - pydantic_ai_langfuse_agent.py:258")
            for agent in (brave_agent, airtable_agent, filesystem_agent, github_agent, slack_agent, firecrawl_agent):
                if getattr(agent, "mcp_servers", None):
                    await stack.enter_async_context(agent.run_mcp_servers())
            print("All configured MCP servers started successfully! - pydantic_ai_langfuse_agent.py:265")
        else:
            print("MCP servers disabled (local-only mode). Running without external MCP servers.")

        console = Console()
        messages = []        
        
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye! - pydantic_ai_langfuse_agent.py:276")
                break
            
            try:
                # Configure the metadata for the Langfuse tracing
                with tracer.start_as_current_span("Pydantic-Ai-Trace") as span:
                    span.set_attribute("langfuse.user.id", "user-456")
                    span.set_attribute("langfuse.session.id", "987654321")

                    # Process the user input and output the response
                    print("\n[Assistant] - pydantic_ai_langfuse_agent.py:286")
                    curr_message = ""
                    with Live('', console=console, vertical_overflow='visible') as live:
                        async with primary_agent.run_stream(
                            user_input, message_history=messages
                        ) as result:
                            async for message in result.stream_text(delta=True):
                                curr_message += message
                                live.update(Markdown(curr_message))
                        
                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())

                    span.set_attribute("input.value", user_input)
                    span.set_attribute("output.value", curr_message)
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)} - pydantic_ai_langfuse_agent.py:303")

if __name__ == "__main__":
    asyncio.run(main())
