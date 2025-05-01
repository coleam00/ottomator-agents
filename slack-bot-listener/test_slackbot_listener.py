#!/usr/bin/env python3
"""
Slack Bot Listener Test

This script tests the SlackBotListener by:
1. Starting the listener for a test channel (configured in .env)
2. Instructing the user to send a test message
3. Checking if the response starts with [MESSAGE_ID]-...

Usage:
    python test_slackbot_listener.py
"""

import os
import asyncio
import logging
import random
import string
import ssl
import certifi
from pathlib import Path
from dotenv import load_dotenv
from slack_sdk.web.async_client import AsyncWebClient
from slackbot_listener import SlackBotListener

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
    logger.error(".env file not found. Please create one based on env.example.")
    exit(1)

# Variable to track if a valid response was received
valid_response_received = False

async def check_slack_connection():
    """
    Check if the Slack API connection is working by making a test API call.
    Logs detailed connection status to help diagnose connection issues.
    """
    logger.info("Checking Slack API connection...")
    
    # Verify required tokens are set
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")
    
    if not bot_token:
        logger.error("❌ ERROR: SLACK_BOT_TOKEN environment variable is not set")
        return False
    
    if not app_token:
        logger.error("❌ ERROR: SLACK_APP_TOKEN environment variable is not set")
        return False
    
    logger.info(f"SLACK_BOT_TOKEN found (starts with: {bot_token[:10]}...)")
    logger.info(f"SLACK_APP_TOKEN found (starts with: {app_token[:10]}...)")
    
    # Create SSL context
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        logger.info(f"SSL context created with certifi CA bundle: {certifi.where()}")
    except Exception as e:
        logger.error(f"❌ Failed to create SSL context: {e}")
        ssl_context = ssl.create_default_context()
        logger.warning("Using default SSL context as fallback")
    
    # Create Slack client and test connection
    try:
        client = AsyncWebClient(token=bot_token, ssl=ssl_context)
        
        # Test API connection by calling auth.test
        logger.info("Testing Slack API with auth.test...")
        response = await client.auth_test()
        
        if response["ok"]:
            logger.info(f"✅ Slack API connection successful!")
            logger.info(f"Connected as: {response['user']} (User ID: {response['user_id']})")
            logger.info(f"Team: {response['team']} (Team ID: {response['team_id']})")
            return True
        else:
            logger.error(f"❌ Slack API connection failed: {response}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to connect to Slack API: {e}", exc_info=True)
        return False

async def test_agent_callback(channel_id, message_text, user_id, thread_ts=None):
    """
    Test callback for the SlackBotListener that checks if messages are being received.
    
    Args:
        channel_id: The Slack channel ID
        message_text: The message text
        user_id: The user who sent the message
        thread_ts: The thread timestamp (for threaded messages)
    """
    global valid_response_received
    
    # Log the message and mark test as successful
    logger.info(f"✅ MESSAGE RECEIVED: '{message_text}' from channel {channel_id}")
    valid_response_received = True
    
    return True

async def main():
    # Check Slack connection first
    connection_ok = await check_slack_connection()
    if not connection_ok:
        logger.error("Slack connection check failed. Please verify your credentials and network connection.")
        print("\n" + "="*80)
        print("❌ SLACK CONNECTION CHECK FAILED")
        print("Please check the following:")
        print("1. Verify your SLACK_BOT_TOKEN and SLACK_APP_TOKEN in the .env file")
        print("2. Ensure your bot has been added to the workspace")
        print("3. Check your internet connection")
        print("4. Verify the bot has the required scopes in your Slack App configuration")
        print("="*80 + "\n")
        return
    
    # Get the test channel from environment
    test_channels = os.getenv("SLACK_LISTEN_CHANNELS")
    if not test_channels:
        logger.error("No test channels configured in SLACK_LISTEN_CHANNELS environment variable")
        return
    
    # Create the Slack bot listener with our test callback
    slack_bot = SlackBotListener(agent_callback=test_agent_callback)
    
    try:
        # Start the bot
        success = await slack_bot.start()
        if not success:
            logger.error("Failed to start Slack bot. Exiting.")
            return
        
        # Print test instructions
        print("\n" + "="*80)
        print("SIMPLE CONNECTION TEST")
        print(f"1. Please send ANY message to one of these channels: {test_channels}")
        print(f"2. The test will exit immediately after receiving a message")
        print(f"3. This test only checks if the bot can receive messages")
        print("="*80 + "\n")
        
        # Run until a message is received or timeout
        print("Waiting for message... (press Ctrl+C to cancel)")
        for i in range(60):
            if valid_response_received:
                print("\n✅ TEST PASSED: Successfully received a message from Slack!")
                break
                
            await asyncio.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f"Still waiting... ({60-i} seconds remaining)")
        
        if not valid_response_received:
            print("\n❌ TEST FAILED: No message was received in 60 seconds")
            print("Make sure you sent a message to the correct channel")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
    finally:
        # Stop the bot
        await slack_bot.stop()
        print("\nTest completed. Slack bot stopped.")

if __name__ == "__main__":
    asyncio.run(main()) 