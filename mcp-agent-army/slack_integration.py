#!/usr/bin/env python3
"""
Slack Integration Layer

Connects the SlackBotListener to the MCP Agent Army's primary agent.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

# Assuming slack_bot_listener.py is importable (e.g., in the same parent directory or installed)
# Adjust the import path if necessary based on your project structure.
# If slack-bot-listener is a sibling directory: from slack_bot_listener.slack_bot_listener import SlackBotListener
# If it's installed: from slack_bot_listener import SlackBotListener
# For now, assume it's discoverable
try:
    from slack_bot_listener import SlackBotListener
except ImportError:
    # Handle cases where the structure might be different
    # This might happen if running scripts from different directories
    try:
        # If mcp-agent-army is the main execution dir, try relative path
        from ..slack_bot_listener.slack_bot_listener import SlackBotListener
    except ImportError:
         logging.error("Could not import SlackBotListener. Ensure the slack-bot-listener directory is accessible.")
         # Define a dummy class to prevent NameErrors later if import fails?
         class SlackBotListener:
             def __init__(self, *args, **kwargs): pass
             async def start(self): pass
             async def stop(self): pass
             def set_monitored_channels(self, *args, **kwargs): pass
             @property
             def is_running(self): return False

# Imports specifically needed for the callback function's interaction with the agent
from pydantic_ai import Agent
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

logger = logging.getLogger(__name__)

# Placeholder functions, will be implemented next
async def process_slack_message_callback(
    channel_id: str,
    message_text: str,
    user_id: str,
    thread_ts: Optional[str],
    agent: Agent, # The primary agent instance
    message_history: List[Dict[str, Any]] # The shared message history
):
    """
    Callback function passed to SlackBotListener.
    Processes a message received from Slack and sends it to the primary agent.
    """
    if not agent:
        logger.error("[Slack Callback] Error: Agent instance is missing.")
        return

    logger.info(f"[Slack Callback] Processing message from channel {channel_id}, user {user_id}, thread_ts: {thread_ts}")

    # Format the query for the agent
    formatted_query = f"""
    Received a message via Slack from user {user_id} in channel {channel_id}:

    "{message_text}"

    Analyze this message and respond appropriately using the available tools.
    If responding via Slack is required, use the correct Slack tool (post or reply).
    """
    if thread_ts:
        formatted_query += f" This message is in a thread (thread_ts='{thread_ts}'). Use slack_reply_to_thread if replying."
    else:
        formatted_query += f" This is a top-level message in channel {channel_id}. Use slack_post_message if posting a new reply."

    # Pass the global history for now. TODO: Implement finer-grained history scoping if needed.
    relevant_history = message_history

    try:
        logger.info(f"[Slack Callback] Sending formatted query to agent: {formatted_query[:100]}...")

        # Use run_stream to potentially display agent's thinking/output in the console
        console = Console()
        with Live('', console=console, vertical_overflow='visible', auto_refresh=False) as live:
            async with agent.run_stream(formatted_query, message_history=relevant_history) as result:
                curr_message = f"[Agent Response for Slack channel {channel_id}]: "
                async for chunk in result.stream_text(delta=True):
                    curr_message += chunk
                    live.update(Markdown(curr_message), refresh=True)
            
            # Update shared message history
            all_messages = result.all_messages()
            if all_messages:
                 message_history.extend(all_messages)
            logger.info(f"[Slack Callback] Agent finished processing for message in {channel_id}.")

    except Exception as e:
        logger.error(f"[Slack Callback] Error running agent for Slack message: {e}", exc_info=True)
        # Consider sending an error message back to Slack here (requires agent call)

async def start_slack_integration(
    agent: Agent,
    message_history: List[Dict[str, Any]]
) -> Tuple[Optional[SlackBotListener], Optional[asyncio.Task]]:
    """
    Initializes and starts the SlackBotListener integration.

    Args:
        agent: The primary MCP Agent instance to use for processing messages.
        message_history: The shared list for storing conversation history.

    Returns:
        A tuple containing the SlackBotListener instance and the asyncio Task
        running it, or (None, None) if initialization fails.
    """
    logger.info("Initializing Slack integration...")
    listener: Optional[SlackBotListener] = None
    listener_task: Optional[asyncio.Task] = None

    try:
        # Define a wrapper for the callback to pass agent and history
        async def wrapped_callback(channel_id: str, message_text: str, user_id: str, thread_ts: Optional[str]):
            await process_slack_message_callback(
                channel_id, message_text, user_id, thread_ts,
                agent=agent, message_history=message_history
            )

        # Instantiate the listener
        listener = SlackBotListener(agent_callback=wrapped_callback)

        # Start the listener task
        logger.info("Starting Slack Bot Listener task...")
        listener_task = asyncio.create_task(listener.start(), name="SlackListenerTask")

        # Short delay to check for immediate startup failure
        await asyncio.sleep(1)
        if listener_task.done():
            try:
                await listener_task # Check for exception
                logger.warning("Slack listener task finished immediately without error?")
            except Exception as e:
                logger.error(f"Slack listener task failed immediately on startup: {e}", exc_info=True)
                return None, None # Failed to start
        else:
            logger.info("Slack Bot Listener task is running.")

        return listener, listener_task

    except ValueError as ve: # Missing env vars
        logger.error(f"Failed to initialize Slack Bot Listener: {ve}")
        return None, None
    except RuntimeError as re: # Slack component init error
        logger.error(f"Failed to initialize Slack Bot Listener components: {re}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during Slack integration startup: {e}", exc_info=True)
        # Clean up task if it was created but failed before returning
        if listener_task and not listener_task.done():
            listener_task.cancel()
        return None, None 