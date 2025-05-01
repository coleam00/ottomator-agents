#!/usr/bin/env python3
"""
Slack Bot Listener Module

This module creates a Slack bot that:
1. Listens to messages in specified channels
2. Forwards messages to the main MCP agent
3. Allows the agent to respond via the slack-mcp-server
"""

import os
import re
import asyncio
import logging
import platform
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.app.async_app import AsyncApp
import ssl
import certifi
from slack_sdk.web.async_client import AsyncWebClient

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


class SlackBotListener:
    """Slack bot that listens to messages in specified channels and forwards them to the agent."""
    
    def __init__(self, agent_callback=None):
        """
        Initialize the Slack bot listener.
        
        Args:
            agent_callback: Function to call when a message is received.
                           Should accept (channel_id, message_text, user_id, thread_ts) as arguments.
        """
        # Record system platform
        self.is_windows = platform.system() == "Windows"
        if self.is_windows:
            logger.info("Running on Windows platform")
        
        # Verify required environment variables
        required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_LISTEN_CHANNELS"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Parse channels to listen to
        self.listen_channels = []
        channels_str = os.getenv("SLACK_LISTEN_CHANNELS", "")
        
        # Handle potential Windows-specific line endings
        channels_str = channels_str.replace('\r', '')
        
        for channel in channels_str.split(","):
            channel = channel.strip()
            if channel:
                # Store both channel IDs and channel names with # prefix
                self.listen_channels.append(channel)
                
        if not self.listen_channels:
            logger.warning("No channels specified in SLACK_LISTEN_CHANNELS. Bot won't listen to any channels.")
        else:
            logger.info(f"Will listen to channels: {', '.join(self.listen_channels)}")
        
        # Initialize the agent callback function
        self.agent_callback = agent_callback
        
        # Create SSL context using certifi
        try:
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
            logger.info(f"Using CA bundle from certifi: {certifi.where()}")
        except Exception as e:
            logger.error(f"Failed to create SSL context using certifi: {e}", exc_info=True)
            self.ssl_context = ssl.create_default_context()
            logger.warning("Falling back to default SSL context.")

        # Initialize the Slack app with properly configured client
        try:
            self.client = AsyncWebClient(
                token=os.environ["SLACK_BOT_TOKEN"],
                ssl=self.ssl_context
            )
            self.app = AsyncApp(client=self.client)
            self.socket_handler = AsyncSocketModeHandler(
                self.app,
                os.environ["SLACK_APP_TOKEN"]
            )
            logger.info("Slack client and app initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Slack client: {e}", exc_info=True)
            raise
        
        # Set up event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for Slack events."""
        
        @self.app.event("message")
        async def handle_message_events(body, say, logger):
            """Handle message events in the channels we're listening to."""
            try:
                event = body.get("event", {})
                
                # Ignore bot messages and messages without text
                if event.get("bot_id") or not event.get("text"):
                    return
                    
                channel_id = event.get("channel")
                user_id = event.get("user")
                message_text = event.get("text", "").strip()
                thread_ts = event.get("thread_ts", event.get("ts"))
                
                # Check if this is a channel we should listen to
                try:
                    channel_info = await self.client.conversations_info(channel=channel_id)
                    channel_name = channel_info.get("channel", {}).get("name", "")
                    
                    logger.debug(f"Received message in channel {channel_id} ({channel_name})")
                    
                    # Check if this channel is in our list (by ID or name with # prefix)
                    if (channel_id in self.listen_channels or 
                        f"#{channel_name}" in self.listen_channels):
                        
                        logger.info(f"Received message in monitored channel {channel_id} ({channel_name}): {message_text[:50]}...")
                        
                        # Forward the message to the agent if a callback is set
                        if self.agent_callback:
                            await self.agent_callback(channel_id, message_text, user_id, thread_ts)
                except Exception as e:
                    logger.error(f"Error processing message in channel {channel_id}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error in message handler: {e}", exc_info=True)
        
        @self.app.event("app_mention")
        async def handle_app_mentions(body, say, logger):
            """Handle app mentions in any channel."""
            try:
                event = body.get("event", {})
                
                channel_id = event.get("channel")
                user_id = event.get("user")
                thread_ts = event.get("thread_ts", event.get("ts"))
                
                # Remove the mention from the text
                full_text = event.get("text", "").strip()
                # Remove <@USER_ID> mentions
                message_text = re.sub(r'<@[A-Z0-9]+>', '', full_text).strip()
                
                if message_text:
                    logger.info(f"Received mention in channel {channel_id}: {message_text[:50]}...")
                    
                    # Always forward mentions to the agent regardless of channel
                    if self.agent_callback:
                        await self.agent_callback(channel_id, message_text, user_id, thread_ts)
            except Exception as e:
                logger.error(f"Error in app_mention handler: {e}", exc_info=True)
    
    async def start(self):
        """Start the Slack bot listener."""
        try:
            logger.info(f"Starting Slack bot listener for channels: {self.listen_channels}")
            
            # First, do a basic connectivity test
            try:
                logger.info("Testing Slack API connection...")
                auth_test = await self.client.auth_test()
                if auth_test["ok"]:
                    logger.info(f"✅ Slack API connection successful! Connected as: {auth_test['user']} in workspace: {auth_test['team']}")
                    
                    # Log the available channels to help with troubleshooting
                    try:
                        logger.info("Fetching list of accessible channels...")
                        channels_response = await self.client.conversations_list(types="public_channel")
                        if channels_response["ok"]:
                            available_channels = [f"#{c['name']} ({c['id']})" for c in channels_response["channels"]]
                            logger.info(f"Available channels: {', '.join(available_channels[:10])}")
                            
                            # Check if our listen channels are in the available channels
                            channel_ids = [c["id"] for c in channels_response["channels"]]
                            channel_names = [f"#{c['name']}" for c in channels_response["channels"]]
                            
                            for channel in self.listen_channels:
                                if channel.startswith('C') and len(channel) >= 9:  # Channel ID format
                                    if channel not in channel_ids:
                                        logger.warning(f"⚠️ Channel ID {channel} is not in the list of accessible channels")
                                else:  # Channel name format
                                    if channel not in channel_names:
                                        logger.warning(f"⚠️ Channel {channel} is not in the list of accessible channels")
                        else:
                            logger.warning(f"Could not fetch channels list: {channels_response.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.warning(f"Error fetching channels list: {e}")
                else:
                    logger.error(f"❌ Slack API connection test failed: {auth_test.get('error', 'Unknown error')}")
                    return False
            except Exception as e:
                logger.error(f"❌ Slack API connection test failed with exception: {e}", exc_info=True)
                logger.error("Please check your SLACK_BOT_TOKEN and internet connection")
                return False
            
            # Start the socket handler
            await self.socket_handler.start_async()
            logger.info("Slack bot listener started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start Slack bot listener: {e}", exc_info=True)
            return False
    
    async def stop(self):
        """Stop the Slack bot listener."""
        try:
            logger.info("Stopping Slack bot listener...")
            await self.socket_handler.close_async()
            logger.info("Slack bot listener stopped")
        except Exception as e:
            logger.error(f"Error stopping Slack bot listener: {e}", exc_info=True)


async def forward_to_agent(channel_id, message_text, user_id, thread_ts=None):
    """
    Example callback function that would forward messages to the agent.
    This should be replaced with your actual implementation.
    
    Args:
        channel_id: The ID of the channel where the message was posted
        message_text: The text content of the message
        user_id: The ID of the user who sent the message
        thread_ts: The thread timestamp (for threaded messages)
    """
    logger.info(f"Would forward to agent - Channel: {channel_id}, Text: {message_text}, User: {user_id}, Thread: {thread_ts}")
    # Implement your agent forwarding logic here
    
    # Example of how you might format a response for the agent to process
    context = {
        "source": "slack",
        "channel_id": channel_id,
        "user_id": user_id,
        "thread_ts": thread_ts,
        "timestamp": datetime.now().isoformat()
    }
    
    # This is where you would send the message to your agent
    # For example: await your_agent.process_message(message_text, context)


# Run this module standalone for testing
async def main():
    # Create the Slack bot listener with our callback
    slack_bot = SlackBotListener(agent_callback=forward_to_agent)
    
    try:
        # Start the bot and keep it running
        success = await slack_bot.start()
        if not success:
            logger.error("Failed to start Slack bot. Exiting.")
            return
            
        logger.info("Slack bot is running. Press Ctrl+C to stop.")
        
        # Keep the bot running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Clean up resources
        await slack_bot.stop()


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main()) 