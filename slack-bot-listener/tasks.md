# Slack Bot Listener Tasks

## Requirements

Based on our discussion, the Slack Bot Listener module needs to fulfill the following requirements:

1.  **Dynamic Channel Listening:** The module must be able to listen to a list of Slack channel IDs provided dynamically by the main application after initialization. It should not rely on environment variables for the initial channel list.
2.  **Process All Messages:** Within the actively monitored channels, the listener must capture and process *every* message sent.
3.  **Callback Mechanism:** When a message is captured in a monitored channel, the listener must invoke a callback function (provided by the main application).
4.  **Context Forwarding:** The callback function must receive essential context about the message, including:
    *   Channel ID
    *   Message Text
    *   User ID
    *   Thread Timestamp (if applicable)
5.  **Enable History Management:** The structure should allow the main application (via the callback) to implement its own conversation history logic based on the forwarded context.
6.  **Socket Mode:** Utilize Slack's Socket Mode for receiving events.
7.  **Token Configuration:** Use environment variables (`SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`) for Slack authentication.

## Tasks

- [X] Create core `SlackBotListener` class structure.
- [X] Implement `__init__` method:
    - Accepts the `agent_callback` function.
    - Reads `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN` from environment variables.
    - Initializes `slack_bolt` `AsyncApp` and `AsyncSocketModeHandler`.
    - Initializes an empty set or list to store active channel IDs.
- [X] Implement a method to update the set of monitored channels (e.g., `set_monitored_channels(channel_ids: set)`).
- [X] Implement `message` event handler:
    - Check if the event's `channel` ID is in the set of monitored channels.
    - If monitored, extract relevant data (text, user, channel, thread_ts).
    - Call the registered `agent_callback` with the extracted data.
- [X] Implement `start` method:
    - Performs necessary checks (e.g., `auth.test`).
    - Connects the `AsyncSocketModeHandler`.
- [X] Implement `stop` method:
    - Gracefully disconnects the `AsyncSocketModeHandler`.
- [X] Add basic logging for key events (start, stop, message received, errors).
- [X] Add error handling for Slack connection issues and during event processing.
- [X] Ensure necessary environment variables are documented (`env.example`).
- [X] Write basic usage instructions (`README.md`).

## Slack Integration Layer Tasks (slack_integration.py)

- [X] Create `slack_integration.py` file.
- [X] Define `process_slack_message_callback` function in `slack_integration.py`:
    - Takes message details, agent instance, and message history list.
    - Formats the query for the agent.
    - Calls `agent.run()` or `agent.run_stream()`.
    - Handles agent response and updates history (or logs).
- [X] Implement `start_slack_integration` async function in `slack_integration.py`:
    - Takes agent instance and message history list as arguments.
    - Defines/wraps the callback function to include the agent/history context.
    - Instantiates `SlackBotListener` with the wrapped callback.

## Refactor `mcp_agent_army.py`

- [X] Import `start_slack_integration` from `slack_integration.py`.
- [X] Remove the local definition of `process_slack_message_for_agent`.
- [X] In `main()`, call `start_slack_integration` after agent creation.
- [X] Update terminal command handling (`listen slack`, etc.) to use the `listener` object returned by the integration function.
- [X] Update the `finally` block to use the `listener` and `listener_task` returned by the integration function for cleanup.