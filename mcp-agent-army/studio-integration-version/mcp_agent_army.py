from __future__ import annotations
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Tuple
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

load_dotenv()

# ========== Helper function to get model configuration ==========
def get_model(model_name=None):
    """
    Get the OpenAI model configuration with optional model override.
    
    Args:
        model_name: Optional override for the model name.
    
    Returns:
        OpenAIModel instance configured with the specified or default model.
    """
    llm = model_name or os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

# ========== Model selection agent and function ==========

# Default model for the model selection agent
MODEL_SELECTOR_DEFAULT = "gpt-3.5-turbo"

# Create the model selection agent
model_selection_agent = Agent(
    get_model(MODEL_SELECTOR_DEFAULT),
    system_prompt="""You are a model selection specialist. Your job is to determine the most appropriate 
    OpenAI model for a given user request based on its complexity, length, and requirements.
    
    For simple queries, factual questions, or basic tasks, select gpt-3.5-turbo.
    For complex reasoning, creative tasks, or detailed analysis, select gpt-4o.
    For medium complexity tasks that need good performance at lower cost, select gpt-4o-mini.
    
    Additionally, rephrase the user's input to be more optimized for the selected model, clarifying ambiguities
    and structuring the query appropriately.
    
    Respond with ONLY a JSON object with two fields:
    1. "model" - The selected model name (e.g., "gpt-3.5-turbo", "gpt-4o-mini", or "gpt-4o")
    2. "rephrased_query" - The optimized version of the user's query
    """
)

async def select_model_for_task(user_query: str) -> Tuple[str, str]:
    """
    Determine the best model and optimize the query for the task.
    
    Args:
        user_query: The original user query.
        
    Returns:
        Tuple of (selected_model_name, rephrased_query).
    """
    try:
        instruction = f"""
        Analyze the following user query and select the most appropriate OpenAI model:
        
        USER QUERY: {user_query}
        
        Based on complexity, determine if this requires:
        - gpt-3.5-turbo (simple queries, factual questions, basic tasks)
        - gpt-4o-mini (medium complexity tasks requiring good performance)
        - gpt-4o (complex reasoning, creative tasks, detailed analysis)
        
        Also rephrase the query to be optimized for the selected model.
        """
        
        result = await model_selection_agent.run(instruction)
        response = result.parse_json_response()
        
        selected_model = response.get("model", os.getenv('MODEL_CHOICE', 'gpt-4o-mini'))
        rephrased_query = response.get("rephrased_query", user_query)
        
        print(f"Selected model: {selected_model}")
        return selected_model, rephrased_query
    except Exception as e:
        print(f"Error in model selection: {e}")
        # Fall back to default model and original query
        return os.getenv('MODEL_CHOICE', 'gpt-4o-mini'), user_query

# ========== Set up MCP servers for each service ==========

# Airtable MCP server
airtable_server = MCPServerStdio(
    'npx', ['-y', 'airtable-mcp-server'],
    env={"AIRTABLE_API_KEY": os.getenv("AIRTABLE_API_KEY")}
)

# Brave Search MCP server
brave_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-brave-search'],
    env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
)

# Filesystem MCP server
filesystem_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-filesystem', os.getenv("LOCAL_FILE_DIR")]
)

# GitHub MCP server
github_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-github'],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")}
)

# Slack MCP server
slack_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-slack'],
    env={
        "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
        "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID")
    }
)

# Firecrawl MCP server
firecrawl_server = MCPServerStdio(
    'npx', ['-y', 'firecrawl-mcp'],
    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")}
)

# ========== Create subagents with their MCP servers ==========

# Airtable agent
airtable_agent = Agent(
    get_model(),
    system_prompt="You are an Airtable specialist. Help users interact with Airtable databases.",
    mcp_servers=[airtable_server]
)

# Brave search agent
brave_agent = Agent(
    get_model(),
    system_prompt="You are a web search specialist using Brave Search. Find relevant information on the web.",
    mcp_servers=[brave_server]
)

# Filesystem agent
filesystem_agent = Agent(
    get_model(),
    system_prompt="You are a filesystem specialist. Help users manage their files and directories.",
    mcp_servers=[filesystem_server]
)

# GitHub agent
github_agent = Agent(
    get_model(),
    system_prompt="You are a GitHub specialist. Help users interact with GitHub repositories and features.",
    mcp_servers=[github_server]
)

# Slack agent
slack_agent = Agent(
    get_model(),
    system_prompt="You are a Slack specialist. Help users interact with Slack workspaces and channels.",
    mcp_servers=[slack_server]
)

# Firecrawl agent
firecrawl_agent = Agent(
    get_model(),
    system_prompt="You are a web crawling specialist. Help users extract data from websites.",
    mcp_servers=[firecrawl_server]
)

# ========== Create the primary orchestration agent ==========
primary_agent = Agent(
    get_model(),
    system_prompt="""You are a primary orchestration agent that can call upon specialized subagents 
    to perform various tasks. Each subagent is an expert in interacting with a specific third-party service.
    Analyze the user request and delegate the work to the appropriate subagent."""
)

# ========== Define tools for the primary agent to call subagents ==========

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
    print(f"Calling Airtable agent with query: {query}")
    result = await airtable_agent.run(query)
    return {"result": result.data}

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
    print(f"Calling Brave agent with query: {query}")
    result = await brave_agent.run(query)
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
    print(f"Calling Filesystem agent with query: {query}")
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
    print(f"Calling GitHub agent with query: {query}")
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
    print(f"Calling Slack agent with query: {query}")
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
    print(f"Calling Firecrawl agent with query: {query}")
    result = await firecrawl_agent.run(query)
    return {"result": result.data}

async def get_mcp_agent_army():
    """
    Initialize and return the primary agent with all MCP servers running.
    This function sets up an AsyncExitStack and starts all MCP servers,
    then returns the primary agent ready to use.
    
    Returns:
        tuple: (primary_agent, stack) - The primary agent and the AsyncExitStack
              that must be kept alive to maintain the MCP server connections
    """
    # Create a new AsyncExitStack that will be returned to the caller
    stack = AsyncExitStack()
    
    # Start all the subagent MCP servers
    print("Starting MCP servers...")
    await stack.enter_async_context(airtable_agent.run_mcp_servers())
    await stack.enter_async_context(brave_agent.run_mcp_servers())
    await stack.enter_async_context(filesystem_agent.run_mcp_servers())
    await stack.enter_async_context(github_agent.run_mcp_servers())
    await stack.enter_async_context(slack_agent.run_mcp_servers())
    await stack.enter_async_context(firecrawl_agent.run_mcp_servers())
    print("All MCP servers started successfully!")
    
    # Return both the primary agent and the stack
    return primary_agent, stack