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

# Import the listener
from slack_bot_listener import SlackBotListener

# Import the integration function
from slack_integration import start_slack_integration

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent, RunContext

load_dotenv()

# ========== Helper function to get model configuration ==========
def get_model():
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

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

# ========== Main execution function ==========

# --- Callback function for Slack Listener ---
# [ This function is now moved to slack_integration.py ]
# async def process_slack_message_for_agent(...): 
#    ...
# --- End Slack Callback ---

async def main():
    """Run the primary agent with a given query and manage Slack listener."""
    print("MCP Agent Army - Multi-agent system using Model Context Protocol")
    print("Enter 'exit' to quit the program.")
    
    # Use AsyncExitStack to manage all MCP servers in one context
    async with AsyncExitStack() as stack:
        # Start all the subagent MCP servers
        print("Starting MCP servers...")
        await stack.enter_async_context(airtable_agent.run_mcp_servers())
        await stack.enter_async_context(brave_agent.run_mcp_servers())
        await stack.enter_async_context(filesystem_agent.run_mcp_servers())
        await stack.enter_async_context(github_agent.run_mcp_servers())
        await stack.enter_async_context(slack_agent.run_mcp_servers())
        await stack.enter_async_context(firecrawl_agent.run_mcp_servers())
        print("All MCP servers started successfully!")

        console = Console()
        messages = []        
        
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
            
            try:
                # Process the user input and output the response
                print("\n[Assistant]")
                with Live('', console=console, vertical_overflow='visible') as live:
                    async with primary_agent.run_stream(
                        user_input, message_history=messages
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))
                    
                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")

            # --- Start Slack Bot Listener if Slack server initialized ---
            if slack_server_started:
                print("Slack MCP Server started. Initializing Slack Bot Listener...")
                # Call the integration function from slack_integration.py
                slack_listener, slack_listener_task = await start_slack_integration(
                    agent=primary_agent,
                    message_history=messages
                )
                if not slack_listener or not slack_listener_task:
                    print("[Error] Slack integration failed to start. Listener will be unavailable.")
                # Note: start_slack_integration handles its own internal errors and logging
            else:
                print("Slack MCP Server failed to start. Slack Bot Listener will not be activated.")
            # ----------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
