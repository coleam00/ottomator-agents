#!/usr/bin/env python
"""
Wrapper script to run the MCP Agent Army in test mode.
This sets the SKIP_MCP_SERVERS environment variable to skip MCP server initialization,
which makes it easier to test the dynamic model selection feature without requiring
API keys for the various services, while still making real API calls to OpenAI.
"""

import os
import sys
import asyncio
import pathlib
import re
from dotenv import load_dotenv, find_dotenv
from mcp_agent_army import main as agent_main

def run_agent_test_mode():
    """Run the MCP Agent Army in test mode."""
    # Get the script directory and project root directory
    script_dir = pathlib.Path(__file__).parent.resolve()
    
    # Load environment variables from .env file with verbose output
    print("Looking for .env file...")
    
    # Try multiple methods to find and load the .env file
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        print("No .env file found in current working directory.")
        # Try to find .env in the script directory explicitly
        dotenv_path = os.path.join(script_dir, '.env')
        if os.path.isfile(dotenv_path):
            print(f"Found .env file at: {dotenv_path}")
        else:
            print(f"No .env file found at: {dotenv_path}")
            # Try one level up
            parent_dotenv = os.path.join(os.path.dirname(script_dir), '.env')
            if os.path.isfile(parent_dotenv):
                print(f"Found .env file in parent directory: {parent_dotenv}")
                dotenv_path = parent_dotenv
            else:
                dotenv_path = None
    else:
        print(f"Found .env file at: {dotenv_path}")
    
    # Load the .env file if found
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
        print("Loaded environment variables from .env file")
        
        # Also try to read the API key directly if environment variables aren't loading
        try:
            with open(dotenv_path, 'r') as env_file:
                env_content = env_file.read()
                # Search for LLM_API_KEY using regex
                api_key_match = re.search(r'LLM_API_KEY\s*=\s*([^\s\n]+)', env_content)
                if api_key_match:
                    extracted_key = api_key_match.group(1).strip()
                    # Remove quotes if present
                    extracted_key = extracted_key.strip('"\'')
                    if extracted_key and extracted_key != "no-api-key-provided":
                        print("Found API key in .env file, setting it directly.")
                        os.environ["LLM_API_KEY"] = extracted_key
        except Exception as e:
            print(f"Error reading .env file directly: {e}")
    else:
        print("WARNING: No .env file found. Using existing environment variables.")
    
    # Set environment variables for test mode - skip MCP servers but use real API calls
    os.environ["SKIP_MCP_SERVERS"] = "true"
    
    # Make sure MODEL_CHOICE is set
    if "MODEL_CHOICE" not in os.environ:
        os.environ["MODEL_CHOICE"] = "gpt-4o-mini"
        print("MODEL_CHOICE not found in environment, setting default: gpt-4o-mini")
    else:
        print(f"Using MODEL_CHOICE from environment: {os.environ['MODEL_CHOICE']}")
    
    # Check if API key is set
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key == "no-api-key-provided":
        print("ERROR: No valid LLM_API_KEY found in environment")
        print("Please ensure your .env file has a valid API key for OpenAI")
        print("Example: LLM_API_KEY=sk-...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        
        # Let the user enter an API key manually
        try:
            manual_key = input("Enter your OpenAI API key (starts with 'sk-'): ")
            if manual_key and manual_key.startswith("sk-"):
                os.environ["LLM_API_KEY"] = manual_key
                print("Using manually entered API key.")
            else:
                print("Invalid API key format or empty key provided.")
                return
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    else:
        # Mask most of the API key for security
        masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "****"
        print(f"Found API key (starts with): {masked_key}")
    
    print("\n=== RUNNING MCP AGENT ARMY IN TEST MODE ===")
    print("MCP servers are disabled for easier testing.")
    print("Making real API calls to OpenAI for model selection with your API key.")
    print("This mode is useful for testing the dynamic model selection feature.")
    print("Commands: 'enable auto model', 'disable auto model', 'exit'")
    print(f"Using default model: {os.getenv('MODEL_CHOICE', 'gpt-4o-mini')}")
    print("================================================\n")
    
    # Run the agent with real model selection
    asyncio.run(agent_main())

if __name__ == "__main__":
    run_agent_test_mode() 