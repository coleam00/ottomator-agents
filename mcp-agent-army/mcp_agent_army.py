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
    
    You have two tasks:
    1. Select the most appropriate model from: "gpt-3.5-turbo", "gpt-4o-mini", or "gpt-4o"
    2. Rephrase the original user query to optimize it for the selected model WITHOUT changing its intent
    
    IMPORTANT: The rephrased query should be a DIRECT replacement for the original query, not an instruction about the query.
    
    Respond with ONLY a JSON object with two fields:
    {
      "model": "the-selected-model-name",
      "rephrased_query": "the optimized version of the EXACT same query"
    }
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
        Here is a user query:
        
        "{user_query}"
        
        Please:
        1. Analyze the complexity of this query
        2. Select the most appropriate model: "gpt-3.5-turbo", "gpt-4o-mini", or "gpt-4o"
        3. Provide a rephrased version of the SAME query optimized for the selected model
        
        IMPORTANT: Do NOT change the intent or meaning of the query. The rephrased query should be a DIRECT replacement
        that the model can use instead of the original, not instructions about the query.
        
        Return ONLY valid JSON in this format:
        {{
          "model": "selected-model-name",
          "rephrased_query": "rephrased-user-query"
        }}
        
        For example, if the user says "what's the weather", your response might be:
        {{
          "model": "gpt-3.5-turbo",
          "rephrased_query": "what is the current weather forecast?"
        }}
        """
        
        result = await model_selection_agent.run(instruction)
        
        # Try to parse the response as JSON
        try:
            # First try using parse_json_response if it exists
            if hasattr(result, 'parse_json_response'):
                response = result.parse_json_response()
            # Otherwise, try parsing the data attribute
            elif hasattr(result, 'data'):
                import json
                response = json.loads(result.data)
            else:
                # Fallback to parsing the string representation
                import json
                response = json.loads(str(result).strip())
                
            selected_model = response.get("model", os.getenv('MODEL_CHOICE', 'gpt-4o-mini'))
            rephrased_query = response.get("rephrased_query", user_query)
            
            # Add a sanity check for the rephrased query
            if "query" in rephrased_query.lower() and "model" in rephrased_query.lower() and "select" in rephrased_query.lower():
                # It looks like the instruction leaked into the rephrased query
                print("Warning: Rephrased query seems to contain instructions. Using original query.")
                rephrased_query = user_query
                
        except (json.JSONDecodeError, AttributeError) as json_err:
            print(f"Error parsing JSON response: {json_err}")
            print(f"Raw response: {result}")
            
            # Try to extract values from the raw response using simple string parsing
            raw_str = str(result)
            if "gpt-3.5-turbo" in raw_str:
                selected_model = "gpt-3.5-turbo"
            elif "gpt-4o-mini" in raw_str:
                selected_model = "gpt-4o-mini"
            elif "gpt-4o" in raw_str:
                selected_model = "gpt-4o"
            else:
                selected_model = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
                
            # Just use the original query if we can't parse the rephrased query
            rephrased_query = user_query
        
        print(f"Selected model: {selected_model}")
        return selected_model, rephrased_query
    except Exception as e:
        print(f"Error in model selection: {str(e)}")
        # If we're in test mode, provide more information about API key issues
        if os.getenv("SKIP_MCP_SERVERS", "").lower() in ("true", "1", "yes"):
            if "api key" in str(e).lower() or "apikey" in str(e).lower():
                print("API key error detected. Make sure your .env file has a valid LLM_API_KEY.")
                print("You can also set the environment variable directly: export LLM_API_KEY=your-key")
            
        # Fall back to default model and original query
        default_model = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
        print(f"Falling back to default model: {default_model}")
        return default_model, user_query

# ========== Set up MCP servers for each service ==========

# Airtable MCP server
airtable_server = MCPServerStdio(
    'npx', ['-y', 'airtable-mcp-server'],
    env={"AIRTABLE_API_KEY": os.getenv("AIRTABLE_API_KEY", "dummy-api-key-for-testing")}
)

# Brave Search MCP server
brave_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-brave-search'],
    env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "dummy-api-key-for-testing")}
)

# Filesystem MCP server
filesystem_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-filesystem', os.getenv("LOCAL_FILE_DIR", "./files")]
)

# GitHub MCP server
github_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-github'],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "dummy-api-key-for-testing")}
)

# Slack MCP server
slack_server = MCPServerStdio(
    'npx', ['-y', '@modelcontextprotocol/server-slack'],
    env={
        "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN", "xoxb-dummy-api-key-for-testing"),
        "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID", "dummy-team-id-for-testing")
    }
)

# Firecrawl MCP server
firecrawl_server = MCPServerStdio(
    'npx', ['-y', 'firecrawl-mcp'],
    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY", "dummy-api-key-for-testing")}
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

async def main():
    """Run the primary agent with a given query."""
    print("MCP Agent Army - Multi-agent system using Model Context Protocol")
    print("Enter 'exit' to quit the program.")
    print("Type 'enable auto model' to enable dynamic model selection.")
    print("Type 'disable auto model' to disable dynamic model selection.")
    
    # Flag to control dynamic model selection
    auto_model_selection = False
    
    # Check if we should skip MCP server startup (useful for testing)
    skip_mcp_servers = os.getenv("SKIP_MCP_SERVERS", "").lower() in ("true", "1", "yes")
    
    if skip_mcp_servers:
        print("Skipping MCP server startup as SKIP_MCP_SERVERS is set.")
        console = Console()
        messages = []
        
        # Simple loop for testing model selection only
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
                
            # Check for auto model commands
            if user_input.lower() == 'enable auto model':
                auto_model_selection = True
                print("[System] Auto model selection enabled! The system will now choose the best model for each task.")
                continue
                
            if user_input.lower() == 'disable auto model':
                auto_model_selection = False
                print("[System] Auto model selection disabled. Using default model for all tasks.")
                continue
            
            try:
                # Process the user input
                print("\n[Assistant]")
                
                # If auto model selection is enabled, determine the best model
                processed_input = user_input
                selected_model = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
                
                if auto_model_selection:
                    print("[System] Selecting optimal model for this task...")
                    selected_model, processed_input = await select_model_for_task(user_input)
                    print(f"[System] Using model: {selected_model}")
                    print(f"[System] Optimized query: {processed_input}")
                    print("[System] MCP servers are disabled, so no actual query processing.")
                else:
                    print("[System] Using default model.")
                    print("[System] MCP servers are disabled, so no actual query processing.")
                
                # Display the model used after the response
                print(f"\n[Model Used: {selected_model}]")
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")
        
        return
    
    # Use AsyncExitStack to manage all MCP servers in one context
    async with AsyncExitStack() as stack:
        # Start all the subagent MCP servers
        print("Starting MCP servers...")
        
        # Define a timeout for MCP server initialization
        MCP_SERVER_TIMEOUT = int(os.getenv("MCP_SERVER_TIMEOUT", "30"))  # Default 30 seconds
        
        # Define which servers are required vs optional
        required_servers = []  # Empty means all are optional
        initialized_servers = []
        server_errors = []
        
        try:
            # Simple wrapper to start a server with error handling
            async def start_single_server(agent, name):
                try:
                    # This is the correct way to enter the context - await it directly
                    await asyncio.wait_for(
                        stack.enter_async_context(agent.run_mcp_servers()),
                        timeout=MCP_SERVER_TIMEOUT
                    )
                    print(f"{name} server started successfully!")
                    initialized_servers.append(name)
                    return True
                except Exception as e:
                    error_msg = f"Error starting {name} server: {str(e)}"
                    print(error_msg)
                    server_errors.append(error_msg)
                    return False
            
            # Start servers one by one with proper error handling
            print("Starting Airtable server...")
            await start_single_server(airtable_agent, "Airtable")
            
            print("Starting Brave Search server...")
            await start_single_server(brave_agent, "Brave Search")
            
            print("Starting Filesystem server...")
            await start_single_server(filesystem_agent, "Filesystem")
            
            print("Starting GitHub server...")
            await start_single_server(github_agent, "GitHub")
            
            print("Starting Slack server...")
            await start_single_server(slack_agent, "Slack")
            
            print("Starting Firecrawl server...")
            await start_single_server(firecrawl_agent, "Firecrawl")
            
            if initialized_servers:
                print(f"Successfully initialized servers: {', '.join(initialized_servers)}")
            else:
                print("Warning: No MCP servers were successfully initialized.")
                print("The agent will continue running but may have limited functionality.")
                if server_errors:
                    print("Server initialization errors:")
                    for error in server_errors:
                        print(f"  - {error}")
            
            print("MCP server initialization completed.")
            
        except Exception as e:
            print(f"Error during MCP server initialization: {str(e)}")
            print("Continuing without MCP servers...")

        console = Console()
        messages = []        
        
        while True:
            # Get user input
            user_input = input("\n[You] ")
            
            # Check if user wants to exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Goodbye!")
                break
                
            # Check for auto model commands
            if user_input.lower() == 'enable auto model':
                auto_model_selection = True
                print("[System] Auto model selection enabled! The system will now choose the best model for each task.")
                continue
                
            if user_input.lower() == 'disable auto model':
                auto_model_selection = False
                print("[System] Auto model selection disabled. Using default model for all tasks.")
                continue
            
            try:
                # Process the user input
                print("\n[Assistant]")
                
                # If auto model selection is enabled, determine the best model
                processed_input = user_input
                selected_model = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
                
                if auto_model_selection:
                    print("[System] Selecting optimal model for this task...")
                    selected_model, processed_input = await select_model_for_task(user_input)
                    # Update the model for the primary agent
                    primary_agent.model = get_model(selected_model)
                    print(f"[System] Using model: {selected_model}")
                    print(f"[System] Optimized query: {processed_input}")
                
                # Process with the selected or default model
                with Live('', console=console, vertical_overflow='visible') as live:
                    async with primary_agent.run_stream(
                        processed_input, message_history=messages
                    ) as result:
                        curr_message = ""
                        async for message in result.stream_text(delta=True):
                            curr_message += message
                            live.update(Markdown(curr_message))
                    
                    # Add the new messages to the chat history
                    messages.extend(result.all_messages())
                
                # Display the model used after the response
                print(f"\n[Model Used: {selected_model}]")
                
                # Reset model to default after processing if auto selection was used
                if auto_model_selection:
                    primary_agent.model = get_model()
                
            except Exception as e:
                print(f"\n[Error] An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
