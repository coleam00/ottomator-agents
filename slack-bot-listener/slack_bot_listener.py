#!/usr/bin/env python3
"""
Slack Bot Listener Module

This module provides a class to listen to specified Slack channels using Socket Mode
and forward messages to a provided callback function.
"""

import os
import asyncio
import logging
import ssl
import certifi
from typing import Callable, Coroutine, Set, Optional

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.app.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define the expected signature for the callback function
# It should accept: channel_id, message_text, user_id, thread_ts
AgentCallbackType = Callable[[str, str, str, Optional[str]], Coroutine[None, None, None]]

class SlackBotListener:
    """Listens to specified Slack channels and forwards messages via a callback."""

    def __init__(self, agent_callback: AgentCallbackType):
        """
        Initialize the Slack bot listener.

        Args:
            agent_callback: Async function to call when a message is received
                           in a monitored channel. Expected signature:
                           async def callback(channel_id: str, message_text: str, user_id: str, thread_ts: Optional[str])

        Raises:
            ValueError: If required environment variables (SLACK_BOT_TOKEN, SLACK_APP_TOKEN) are missing.
            RuntimeError: If core Slack components fail to initialize.
        """
        logger.debug("Initializing SlackBotListener...")
        self.agent_callback = agent_callback
        if not asyncio.iscoroutinefunction(agent_callback):
            logger.warning("Provided agent_callback is not an async function. Ensure it behaves correctly.")

        self.monitored_channels: Set[str] = set()
        self._is_running = False # Internal state flag
        self._start_lock = asyncio.Lock() # Prevent concurrent start/stop operations

        # --- Slack App Initialization ---
        try:
            # Retrieve tokens from environment variables
            self.bot_token = os.environ["SLACK_BOT_TOKEN"]
            self.app_token = os.environ["SLACK_APP_TOKEN"]
            logger.info("Slack tokens retrieved from environment.")
        except KeyError as e:
            logger.error(f"Missing required environment variable: {e}. Cannot initialize SlackBotListener.")
            raise ValueError(f"Missing required environment variable: {e}") from e

        # Create SSL context using certifi
        try:
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
            logger.debug(f"Using CA bundle from certifi: {certifi.where()}")
        except Exception as e:
            # Log error but fallback gracefully
            logger.error(f"Failed to create SSL context using certifi: {e}", exc_info=True)
            self.ssl_context = ssl.create_default_context()
            logger.warning("Falling back to default system SSL context.")

        # Initialize the Slack app with a properly configured client
        try:
            self.client = AsyncWebClient(
                token=self.bot_token,
                ssl=self.ssl_context
            )
            self.app = AsyncApp(client=self.client)
            self.socket_handler = AsyncSocketModeHandler(
                self.app,
                self.app_token
                # logger can be configured here if needed: logger=custom_logger
            )
            logger.info("Slack client, app, and socket handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Slack client/app/handler: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize core Slack components") from e

        # Register event handlers
        self._setup_event_handlers()
        logger.debug("Slack event handlers registered.")


    def _setup_event_handlers(self):
        """Set up event handlers for Slack events. Called during init."""

        @self.app.event("message")
        async def handle_message_events(body: dict, say: Callable, logger_prop):
            """
            Handle message events, forwarding relevant ones to the agent callback.
            Uses logger_prop from slack_bolt, not the module-level logger.
            """
            # Use the logger provided by slack_bolt for event-specific logging
            event_logger = logger_prop or logger # Fallback to module logger if needed
            try:
                event = body.get("event", {})
                event_logger.debug(f"Received event body: {body}")

                # Ignore subtypes like channel_join, message_deleted etc. Check for actual message text.
                # Also ignore messages from bots (including self), and messages without text or user
                message_subtype = event.get("subtype")
                if message_subtype is not None and message_subtype != "thread_broadcast":
                    event_logger.debug(f"Ignoring message with subtype: {message_subtype}")
                    return
                if event.get("bot_id") or not event.get("text") or not event.get("user"):
                    event_logger.debug(f"Ignoring non-user message or message without text/user.")
                    return

                channel_id = event.get("channel")
                user_id = event.get("user")
                message_text = event.get("text", "").strip()
                thread_ts = event.get("thread_ts", None) # Explicitly None if not a thread reply
                message_ts = event.get("ts") # The actual timestamp of this specific message

                if not channel_id or not message_ts:
                    event_logger.warning("Received message event without channel_id or ts.")
                    return

                # Check if this message is in a channel we are monitoring
                if channel_id in self.monitored_channels:
                    event_logger.info(f"Received message in monitored channel {channel_id} (ts:{message_ts}): {message_text[:50]}...")

                    # Call the registered agent callback function
                    try:
                        await self.agent_callback(
                            channel_id=channel_id,
                            message_text=message_text,
                            user_id=user_id,
                            thread_ts=thread_ts # Pass the thread_ts for context
                        )
                        event_logger.debug(f"Agent callback successfully executed for message {message_ts}.")
                    except asyncio.CancelledError:
                        event_logger.warning(f"Agent callback task cancelled for message {message_ts}.")
                        # Re-raise cancellation if necessary for upstream handling
                        raise
                    except Exception as callback_err:
                        event_logger.error(f"Error executing agent_callback for message {message_ts} in channel {channel_id}: {callback_err}", exc_info=True)
                        # Consider sending an error notification back to Slack? Needs careful implementation.
                        # Example (disabled): await say(text=f"Sorry <@{user_id}>, I encountered an error processing your message.", thread_ts=thread_ts)
                else:
                    # Log messages in non-monitored channels at debug level
                    event_logger.debug(f"Ignoring message in non-monitored channel {channel_id} (ts:{message_ts})")

            except Exception as e:
                # Log errors using the logger passed by slack_bolt
                event_logger.error(f"Unhandled error in handle_message_events: {e}", exc_info=True)

    # --- Methods for managing monitored channels ---
    def set_monitored_channels(self, channel_ids: Set[str]):
        """
        Update the set of channel IDs the listener should monitor.

        Args:
            channel_ids: A set of Slack channel IDs.
        """
        # Ensure input is a set for efficient lookup
        if not isinstance(channel_ids, set):
            logger.warning(f"Received non-set type for channel_ids: {type(channel_ids)}. Attempting conversion.")
            try:
                channel_ids = set(channel_ids)
            except TypeError:
                logger.error(f"Could not convert provided channel_ids to set. Keeping previous value: {self.monitored_channels}")
                return

        logger.info(f"Updating monitored channels from {self.monitored_channels} to {channel_ids}")
        # This update should be thread/task safe if modified by multiple contexts, consider locks if needed.
        self.monitored_channels = channel_ids
        logger.debug(f"Currently monitoring channels: {self.monitored_channels}")

    # --- Methods for starting and stopping the listener ---
    @property
    def is_running(self) -> bool:
        """Returns True if the listener is currently active, False otherwise."""
        # Check both the internal flag and the socket handler's connection status
        return self._is_running and self.socket_handler and self.socket_handler.client.is_connected()

    async def start(self) -> bool:
        """
        Start the Slack bot listener.

        Performs an authentication test and connects the Socket Mode handler.
        This method will run until the listener is stopped or encounters a fatal error.

        Returns:
            True if startup sequence initiated successfully, False if checks fail.
            Note: The method blocks until stopped. Success means it ran without crashing immediately.
        """
        async with self._start_lock:
            if self._is_running:
                logger.warning("Listener start requested, but it is already running.")
                return True

            logger.info("Attempting to start Slack bot listener...")
            try:
                # Perform an auth test to verify token and connectivity
                logger.info("Testing Slack API connection (auth.test)...")
                auth_response = await self.client.auth_test()
                # Expected keys: 'ok', 'url', 'team', 'user', 'team_id', 'user_id', 'bot_id', 'is_enterprise_install'
                if auth_response.get("ok"):
                    logger.info(f"✅ Slack auth.test successful! Bot ID: {auth_response.get('bot_id')}, User ID: {auth_response.get('user_id')} in Team: {auth_response.get('team')} ({auth_response.get('team_id')})")
                else:
                    error_detail = auth_response.get("error", "Unknown error")
                    logger.error(f"❌ Slack auth.test failed: {error_detail}")
                    logger.error("Please check your SLACK_BOT_TOKEN and required permissions.")
                    return False # Failed initial check
            except SlackApiError as e:
                # Handle common API errors more specifically if possible
                error_code = e.response.get("error", "unknown_api_error")
                logger.error(f"❌ Slack API error during auth.test ({error_code}): {e}", exc_info=True)
                logger.error("Check network connectivity, SLACK_BOT_TOKEN validity, and bot permissions (auth.test scope).")
                return False
            except Exception as e:
                logger.error(f"❌ Unexpected error during Slack auth.test: {e}", exc_info=True)
                return False

            # Connect the Socket Mode handler
            try:
                self._is_running = True # Set running state before blocking call
                logger.info("Connecting Socket Mode handler... (This will run until stopped)")
                await self.socket_handler.start_async()
                # If start_async returns, it means it was stopped gracefully or encountered an error handled internally
                logger.info("Socket Mode handler has stopped.")
                self._is_running = False
                return True # If it ran and stopped without throwing an exception here, consider it successful

            except ConnectionRefusedError as e:
                logger.error(f"❌ Connection refused when starting Socket Mode handler: {e}")
                logger.error("Is the Slack API endpoint reachable? Check firewalls or network issues.")
                self._is_running = False
                return False
            except asyncio.CancelledError:
                logger.info("Socket Mode handler task cancelled.")
                self._is_running = False
                # Re-raise if cancellation needs to propagate
                raise
            except Exception as e:
                logger.error(f"❌ Socket Mode handler failed unexpectedly: {e}", exc_info=True)
                self._is_running = False
                # Depending on the error, returning False might be appropriate
                # Or re-raising if it's unrecoverable
                return False

    async def stop(self):
        """Stop the Slack bot listener gracefully."""
        async with self._start_lock:
            if not self._is_running and (not self.socket_handler or not self.socket_handler.client.is_connected()):
                logger.warning("Listener stop requested, but it is not running or already disconnected.")
                return

            logger.info("Attempting to stop Slack bot listener...")
            try:
                if self.socket_handler and self.socket_handler.client.is_connected():
                    # close_async signals the handler to stop accepting new connections and disconnect
                    await self.socket_handler.close_async()
                    logger.info("Socket Mode handler close signal sent.")
                else:
                    logger.warning("Socket handler not connected, cannot send close signal.")

                self._is_running = False
                logger.info("Slack bot listener state set to stopped.")

            except asyncio.CancelledError:
                 logger.warning("Stop operation cancelled.")
                 self._is_running = False # Ensure state is consistent
                 raise # Propagate cancellation
            except Exception as e:
                logger.error(f"Error during Slack bot listener stop: {e}", exc_info=True)
                # Even if stopping fails, reflect the intent to stop in the state
                self._is_running = False

# Example usage (for testing purposes)
async def example_callback(channel_id: str, message_text: str, user_id: str, thread_ts: Optional[str]):
    logger.info(f"[CALLBACK] Received: Channel={channel_id}, User={user_id}, Thread={thread_ts}, Text='{message_text[:30]}...'')
    await asyncio.sleep(0.1) # Simulate some async work

async def run_test():
    # Ensure environment variables are set before running this
    try:
        listener = SlackBotListener(agent_callback=example_callback)
        listener.set_monitored_channels({"YOUR_TEST_CHANNEL_ID"}) # Replace with a real channel ID for testing

        # Start the listener in a separate task
        listener_task = asyncio.create_task(listener.start(), name="SlackListenerTestTask")

        # Keep running for a while, then stop
        await asyncio.sleep(60) # Listen for 60 seconds

        logger.info("Requesting listener stop...")
        await listener.stop()

        # Wait for the listener task to finish
        logger.info("Waiting for listener task to complete...")
        await listener_task
        logger.info("Listener task completed.")

    except ValueError as e: # Catch missing env vars
        logger.error(f"Setup failed: {e}")
    except Exception as e:
        logger.error(f"Test run failed: {e}", exc_info=True)

if __name__ == "__main__":
    # This basic test requires SLACK_BOT_TOKEN and SLACK_APP_TOKEN environment variables
    # and a valid channel ID to monitor.
    logger.info("Running basic SlackBotListener test...")
    # asyncio.run(run_test()) # Ensure this line is correctly commented or formed if uncommented
    logger.info("Test run finished (or uncomment asyncio.run to execute).")

