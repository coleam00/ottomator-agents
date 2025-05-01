#!/usr/bin/env python3
"""
Slack Agent Integration Module

This module integrates the SlackBotListener with the MCP Agent system:
1. Initializes the MCP server for Slack
2. Sets up the Slack bot listener
3. Handles message forwarding between systems
4. Manages the agent's responses back to Slack
"""

import os
import asyncio
import logging
import platform
import time
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from slackbot_listener import SlackBotListener

# For the MCP Agent integration
# Import these according to your actual MCP agent implementation
try:
    # Try to import the necessary MCP components
    # These imports assume your MCP agent setup follows a similar structure to examples
    from mcp_agent_army import (
        get_model, Agent, MCPServerStdio
    )
    MCP_IMPORTS_AVAILABLE = True
except ImportError:
    logging.warning("mcp_agent_army imports not available. Will try direct import.")
    try:
        # Try alternative import paths
        from agents import Agent
        from agents.mcp import MCPServerStdio
        
        # Define a simple get_model function if not available
        def get_model():
            return os.getenv("MODEL_NAME", "gpt-4")
            
        MCP_IMPORTS_AVAILABLE = True
    except ImportError:
        logging.warning("Alternative MCP agent imports not available. Running in standalone mode.")
        MCP_IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info("Loaded environment variables from .env file")
else:
    logger.warning(".env file not found. Make sure to set environment variables manually.")

# Check for Node.js and NPX
def check_nodejs_availability():
    """Check if Node.js and NPX are available on the system."""
    is_windows = platform.system() == "Windows"
    logger.info(f"Operating system: {platform.system()} {platform.release()}")
    
    # First, try to find node and npx in PATH
    node_path = shutil.which('node')
    npx_path = shutil.which('npx')
    
    if node_path:
        logger.info(f"Node.js found at: {node_path}")
    else:
        logger.error("Node.js not found in PATH")
        
    if npx_path:
        logger.info(f"NPX found at: {npx_path}")
    else:
        logger.error("NPX not found in PATH")
    
    try:
        # Check Node.js
        node_cmd = "node --version"
        logger.debug(f"Running command: {node_cmd}")
        node_result = subprocess.run(node_cmd, shell=True, capture_output=True, text=True)
        if node_result.returncode == 0:
            logger.info(f"Node.js version: {node_result.stdout.strip()}")
        else:
            logger.error(f"Node.js not found: {node_result.stderr.strip()}")
            return False
            
        # Check NPX
        npx_cmd = "npx --version"
        logger.debug(f"Running command: {npx_cmd}")
        npx_result = subprocess.run(npx_cmd, shell=True, capture_output=True, text=True)
        if npx_result.returncode == 0:
            logger.info(f"NPX version: {npx_result.stdout.strip()}")
        else:
            logger.error(f"NPX not found: {npx_result.stderr.strip()}")
            return False
            
        # Try to find npx in common Windows locations
        if is_windows:
            potential_paths = [
                os.path.join(os.environ.get('APPDATA', ''), 'npm'),
                os.path.join(os.environ.get('ProgramFiles', ''), 'nodejs'),
                os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'nodejs'),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'npm')
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    logger.info(f"Found potential Node.js path: {path}")
                    if os.path.exists(os.path.join(path, 'npx.cmd')) or os.path.exists(os.path.join(path, 'npx')):
                        npx_path = os.path.join(path, 'npx.cmd')
                        logger.info(f"Found npx at: {npx_path}")
                        # Set this in environment for later use
                        os.environ['NPX_PATH'] = npx_path
        
        # Test npx capability by running a simple command
        test_cmd = "npx --no-install --version"
        logger.debug(f"Testing npx functionality: {test_cmd}")
        test_result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
        if test_result.returncode == 0:
            logger.info("NPX functionality test passed")
        else:
            logger.warning(f"NPX functionality test failed: {test_result.stderr.strip()}")
            
        return True
    except Exception as e:
        logger.error(f"Error checking Node.js environment: {e}", exc_info=True)
        return False


class SlackAgentIntegration:
    """Integrates the Slack bot listener with the MCP agent system."""
    
    def __init__(self):
        """Initialize the integration components."""
        # Record system platform
        self.is_windows = platform.system() == "Windows"
        if self.is_windows:
            logger.info("Running on Windows platform")
            
        # Check for required environment variables
        required_vars = [
            "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_TEAM_ID", "SLACK_LISTEN_CHANNELS"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Initialize components
        self.slack_mcp_server = None
        self.agent = None
        self.slack_bot = None
        self.running = False
        
    async def initialize(self):
        """Initialize the MCP server, agent, and Slack bot."""
        try:
            # First verify that Node.js and NPX are available
            logger.info("Checking Node.js and NPX availability...")
            if not check_nodejs_availability():
                logger.error("Node.js or NPX not available, cannot initialize MCP server")
                return False
            
            # Verify Slack API connection before proceeding
            logger.info("Checking Slack API connection...")
            try:
                import ssl
                import certifi
                from slack_sdk.web.async_client import AsyncWebClient
                
                # Create SSL context
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                
                # Create temporary client to test connection
                bot_token = os.getenv("SLACK_BOT_TOKEN")
                if not bot_token:
                    logger.error("❌ SLACK_BOT_TOKEN environment variable is not set")
                    return False
                
                client = AsyncWebClient(token=bot_token, ssl=ssl_context)
                
                # Test API connection
                auth_response = await client.auth_test()
                if auth_response["ok"]:
                    logger.info(f"✅ Slack API connection successful!")
                    logger.info(f"Connected as: {auth_response['user']} (User ID: {auth_response['user_id']})")
                    logger.info(f"Team: {auth_response['team']} (Team ID: {auth_response['team_id']})")
                    
                    # Check if team ID matches configuration
                    expected_team_id = os.getenv("SLACK_TEAM_ID")
                    if expected_team_id and expected_team_id != auth_response["team_id"]:
                        logger.warning(f"⚠️ SLACK_TEAM_ID in .env ({expected_team_id}) doesn't match the actual team ID ({auth_response['team_id']})")
                else:
                    logger.error(f"❌ Slack API connection failed: {auth_response.get('error', 'Unknown error')}")
                    return False
            except Exception as e:
                logger.error(f"❌ Failed to connect to Slack API: {e}", exc_info=True)
                logger.error("Please verify your SLACK_BOT_TOKEN and internet connection")
                return False
                
            # Step 1: Initialize the Slack MCP server
            if MCP_IMPORTS_AVAILABLE:
                logger.info("Initializing Slack MCP server...")
                self.slack_mcp_server = await self._create_slack_mcp_server()
                if not self.slack_mcp_server:
                    logger.error("Failed to create Slack MCP server")
                    return False
                
                # Step 2: Initialize the agent with access to the MCP server
                logger.info("Initializing MCP agent...")
                self.agent = await self._create_agent(self.slack_mcp_server)
                if not self.agent:
                    logger.error("Failed to create agent")
                    return False
            else:
                logger.warning("Running without MCP components. Agent functionality will be limited.")
            
            # Step 3: Initialize the Slack bot listener
            logger.info("Initializing Slack bot listener...")
            self.slack_bot = SlackBotListener(agent_callback=self.process_slack_message)
            
            logger.info("Initialization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False
        
    async def _create_slack_mcp_server(self):
        """Create and return a Slack MCP server instance."""
        if not MCP_IMPORTS_AVAILABLE:
            return None
            
        try:
            logger.info("Creating Slack MCP server...")
            
            # Determine command and args based on platform
            cmd = 'npx'
            npx_args = ['-y', '@modelcontextprotocol/server-slack']
            
            # Special handling for Windows
            if self.is_windows:
                logger.info("Using Windows-specific configuration for MCP server")
                
                # Check if we found a specific path to NPX earlier
                if 'NPX_PATH' in os.environ:
                    cmd = os.environ['NPX_PATH']
                    logger.info(f"Using NPX from detected path: {cmd}")
                
                # Optional: Check if npx is available
                try:
                    # This will throw an exception if npx is not found
                    import subprocess
                    result = subprocess.run(["npx", "--version"], 
                                  check=True, 
                                  capture_output=True, 
                                  shell=True)
                    logger.info(f"NPX version: {result.stdout.decode().strip()}")
                except Exception as e:
                    logger.warning(f"npx check failed: {e}, but continuing anyway")
            
            # Log environment variables being passed to the server (without exposing the full token)
            slack_token = os.getenv("SLACK_BOT_TOKEN", "")
            logger.info(f"Using SLACK_BOT_TOKEN: {'xoxb-...' if slack_token.startswith('xoxb-') else 'Invalid token'}")
            slack_team_id = os.getenv("SLACK_TEAM_ID", "")
            logger.info(f"Using SLACK_TEAM_ID: {slack_team_id}")
                
            # Set up the environment with needed variables
            server_env = {
                "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
                "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID")
            }
            
            # Create the Slack MCP server
            logger.info(f"Initializing MCPServerStdio with cmd: {cmd}, args: {npx_args}")
            slack_server = MCPServerStdio(
                cmd, npx_args,
                env=server_env
            )
            
            # Give the server some time to initialize
            logger.info("Waiting for MCP server initialization...")
            await asyncio.sleep(5)
            
            logger.info("Slack MCP server created successfully")
            return slack_server
        except Exception as e:
            logger.error(f"Failed to create Slack MCP server: {e}", exc_info=True)
            return None
        
    async def _create_agent(self, slack_server):
        """Create and return an agent with the Slack MCP server."""
        if not MCP_IMPORTS_AVAILABLE or not slack_server:
            return None
            
        try:
            # Create the agent with access to the Slack MCP server
            logger.info("Creating agent with Slack MCP server")
            agent = Agent(
                get_model(),
                system_prompt="""You are a Slack assistant that monitors channels and responds to messages.
                Use the Slack MCP server to interact with Slack, responding in a helpful and concise manner.
                You can post messages, reply to threads, and retrieve information as needed.""",
                mcp_servers=[slack_server]
            )
            logger.info("Agent created successfully")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent: {e}", exc_info=True)
            return None
        
    async def process_slack_message(self, channel_id, message_text, user_id, thread_ts=None):
        """
        Process a message from Slack and forward it to the agent.
        
        Args:
            channel_id: The Slack channel ID
            message_text: The message text
            user_id: The user who sent the message
            thread_ts: The thread timestamp (for threaded messages)
        """
        if not self.agent:
            logger.warning("Agent not available. Cannot process message.")
            return
            
        logger.info(f"Processing message from channel {channel_id}")
        
        # Create a context dictionary for the agent
        context = {
            "channel_id": channel_id,
            "user_id": user_id,
            "thread_ts": thread_ts
        }
        
        # Format the query for the agent, including context information
        formatted_query = f"""
        Message from Slack channel {channel_id}:
        
        {message_text}
        
        Please respond to this message. Use the slack_post_message or slack_reply_to_thread tool 
        to send your response back to the channel {channel_id}."""
        
        # Add thread context if applicable
        if thread_ts:
            formatted_query += f" This is a threaded message, so please use slack_reply_to_thread with thread_ts={thread_ts}."
        
        # Process the message with the agent
        try:
            # Run the agent with the message
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Run the agent with the message
                    logger.info(f"Sending message to agent (attempt {retry_count + 1}/{max_retries})")
                    result = await self.agent.run(formatted_query)
                    logger.info(f"Agent processed message. Response: {result.data[:100]}...")
                    return
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error processing message with agent (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        # Wait briefly before retrying
                        await asyncio.sleep(2)
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Error processing message with agent: {e}", exc_info=True)
            
    async def start(self):
        """Start the integration."""
        if self.running:
            logger.warning("Integration is already running.")
            return False
            
        try:
            # Initialize components if not already done
            if not self.slack_bot:
                success = await self.initialize()
                if not success:
                    logger.error("Initialization failed")
                    return False
                
            # Start the Slack bot listener
            success = await self.slack_bot.start()
            if not success:
                logger.error("Failed to start Slack bot listener")
                return False
                
            self.running = True
            logger.info("Slack agent integration started successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to start integration: {e}", exc_info=True)
            return False
        
    async def stop(self):
        """Stop the integration."""
        if not self.running:
            logger.warning("Integration is not running.")
            return
            
        try:
            # Stop the Slack bot listener
            if self.slack_bot:
                await self.slack_bot.stop()
                
            self.running = False
            logger.info("Slack agent integration stopped.")
        except Exception as e:
            logger.error(f"Error stopping integration: {e}", exc_info=True)
        

# Run this module standalone
async def main():
    # Create and start the integration
    integration = SlackAgentIntegration()
    
    try:
        # Initialize and start the integration
        success = await integration.start()
        if not success:
            logger.error("Failed to start integration. Exiting.")
            return
            
        logger.info("Slack agent integration is running. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Clean up resources
        await integration.stop()


if __name__ == "__main__":
    # Run the integration
    print("Starting Slack agent integration...")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nIntegration stopped by user")
        sys.exit(0) 