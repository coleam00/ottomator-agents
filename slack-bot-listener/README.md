# Slack Bot Listener Module

This module provides a Python class (`SlackBotListener`) to listen for messages in specific Slack channels using Socket Mode and forward relevant message details to a provided asynchronous callback function.

## Features

- **Socket Mode Connection:** Uses Slack's Socket Mode for efficient event handling.
- **Dynamic Channel Monitoring:** Allows the main application to specify which channels to monitor *after* initialization.
- **Callback Mechanism:** Forwards message details (channel, user, text, thread context) to a user-defined async callback function for processing.
- **Environment Variable Configuration:** Relies on standard environment variables for Slack tokens.
- **Basic Error Handling:** Includes checks for missing tokens and handles common connection errors.

## Prerequisites

- Python 3.7+
- A Slack workspace where you can create and install a Slack App.

## Setup

### 1. Create a Slack App for Socket Mode

1.  Go to [https://api.slack.com/apps](https://api.slack.com/apps).
2.  Click "Create New App" -> "From scratch".
3.  Give your app a name and select your workspace.
4.  Navigate to "Socket Mode" under "Features" and enable it.
5.  Generate an **App-Level Token** (starts with `xapp-`) with the `connections:write` scope. Copy this token.
6.  Navigate to "OAuth & Permissions" under "Features".
7.  Add the following **Bot Token Scopes**:
    *   `channels:history` (to read messages in public channels)
    *   `groups:history` (to read messages in private channels)
    *   `im:history` (to read messages in DMs)
    *   `mpim:history` (to read messages in group DMs)
    *   `users:read` (optional, helps resolve user IDs)
    *   `auth.test` (required for basic connection verification)
    *   Add any scopes your *callback* function might need (e.g., `chat:write` if your callback needs to send messages).
8.  Install the app to your workspace.
9.  Copy the **Bot User OAuth Token** (starts with `xoxb-`).

### 2. Environment Variables

Create a `.env` file in your project root (or set environment variables directly) with the tokens obtained above:

```dotenv
# .env
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
```

### 3. Install Dependencies

```bash
pip install slack-bolt python-dotenv certifi
```

## Usage

The `SlackBotListener` class is designed to be integrated into an asynchronous Python application.

```python
import asyncio
import os
import logging
from slack_bot_listener import SlackBotListener # Assuming the file is accessible
from typing import Optional, Set

# Load environment variables (ensure SLACK_BOT_TOKEN and SLACK_APP_TOKEN are set)
from dotenv import load_dotenv
load_dotenv()

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# 1. Define your asynchronous callback function
# This function will be called by the listener when a message is received in a monitored channel.
async def my_message_processor(channel_id: str, message_text: str, user_id: str, thread_ts: Optional[str]):
    logging.info(f"[My Processor] Received in {channel_id} from {user_id}: '{message_text[:30]}...' Thread: {thread_ts}")
    # --- Add your application's logic here --- 
    # For example, send this data to your main AI agent, manage history, etc.
    await asyncio.sleep(0.1) # Simulate processing


async def run_application():
    listener = None # Keep track of the listener instance
    listener_task = None # Keep track of the running task
    try:
        # 2. Initialize the listener with your callback
        listener = SlackBotListener(agent_callback=my_message_processor)

        # 3. Tell the listener which channels to monitor initially (can be updated later)
        # Replace with actual channel IDs obtained from Slack (e.g., C12345678)
        initial_channels: Set[str] = {"YOUR_CHANNEL_ID_1"}
        listener.set_monitored_channels(initial_channels)

        # --- Your Application's Main Logic --- 
        # Maybe start other services, run a web server, etc.
        logging.info("Application started. Starting Slack listener...")

        # 4. Start the listener in a background task
        # The listener.start() method blocks until stopped, so run it concurrently.
        listener_task = asyncio.create_task(listener.start(), name="SlackListenerTask")

        # --- Keep your main application alive --- 
        # Example: Run indefinitely until interrupted
        while True:
            # You could add logic here to dynamically update monitored channels:
            # if should_add_channel:
            #    new_channel_set = listener.monitored_channels.union({"NEW_CHANNEL_ID"})
            #    listener.set_monitored_channels(new_channel_set)
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logging.info("Application received cancellation signal.")
    except ValueError as e: # Catches missing env vars from listener init
        logging.error(f"Configuration Error: {e}")
    except RuntimeError as e: # Catches Slack component init errors
        logging.error(f"Runtime Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main application: {e}", exc_info=True)
    finally:
        # 5. Stop the listener gracefully when your application exits
        if listener and listener_task and not listener_task.done():
            logging.info("Stopping Slack listener...")
            await listener.stop() # Request graceful shutdown
            try:
                await asyncio.wait_for(listener_task, timeout=5.0) # Wait for task to finish
                logging.info("Listener task finished.")
            except asyncio.TimeoutError:
                logging.warning("Listener task did not finish within timeout after stop request.")
            except asyncio.CancelledError:
                logging.info("Listener task was cancelled during shutdown.")
        elif listener_task and listener_task.done():
             logging.info("Listener task already completed.")

        logging.info("Application shutdown complete.")

if __name__ == "__main__":
    # To run this example:
    # 1. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN in your environment or a .env file
    # 2. Replace "YOUR_CHANNEL_ID_1" with an actual channel ID where the bot is present
    # 3. Run the script (e.g., python your_main_app.py)
    try:
        asyncio.run(run_application())
    except KeyboardInterrupt:
        logging.info("Shutdown requested via KeyboardInterrupt.")
```

## How it Works

- The `SlackBotListener` uses `slack_bolt`'s `AsyncSocketModeHandler` to establish a persistent WebSocket connection with Slack.
- When a `message` event arrives, the listener checks if the message's channel ID is in the `monitored_channels` set.
- If the channel is monitored and the message is from a user (not a bot or subtype), it calls the `agent_callback` function you provided, passing the relevant message details.
- The `start()` method performs an `auth.test` and then runs the `AsyncSocketModeHandler`, blocking until `stop()` is called or an error occurs.
- The `stop()` method gracefully closes the WebSocket connection.
- The `set_monitored_channels()` method allows the controlling application to dynamically change which channels are being listened to.

## Error Handling

The listener includes basic error handling for:
- Missing environment variables during initialization.
- Failures during Slack client/handler initialization.
- `auth.test` failures during startup.
- Connection errors when starting the Socket Mode handler.
- Errors within the message event handler.
- Errors during the callback execution are logged, but the listener continues running.
