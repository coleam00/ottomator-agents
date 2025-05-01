# Slack Bot Listener for MCP Agent

This project integrates a Slack bot with an MCP agent to create a system that can:
1. Listen to messages in specified Slack channels
2. Forward those messages to an MCP agent for processing
3. Allow the agent to respond directly back to Slack channels

## Features

- **Channel Monitoring**: Listen to specific channels defined in your configuration
- **Mention Support**: Automatically respond to @mentions in any channel
- **Thread Support**: Handle threaded conversations
- **MCP Integration**: Seamless integration with the existing slack-mcp-server
- **Environment Configuration**: Simple setup through environment variables

## Prerequisites

- Python 3.7+
- A Slack workspace with admin privileges
- Node.js and NPX (for the MCP server)
- An existing MCP agent setup (optional but recommended)

## Setup

### 1. Create a Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" and choose "From scratch"
3. Give your app a name and select your workspace
4. In the "Add features and functionality" section:
   - Enable Socket Mode (requires an app-level token)
   - Add Bot Token Scopes: `channels:history`, `channels:read`, `chat:write`, `reactions:write`, `users:read`, `app_mentions:read`
   - Enable Event Subscriptions and subscribe to bot events:
     - `message.channels`
     - `message.groups`
     - `message.im`
     - `app_mention`
5. Install the app to your workspace
6. Copy the "Bot User OAuth Token" (starts with `xoxb-`)
7. Generate an app-level token with the `connections:write` scope
8. Copy the app-level token (starts with `xapp-`)

### 2. Environment Configuration

Copy the `.env.example` file to `.env` and fill in the necessary values:

```
# Slack API Credentials
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_TEAM_ID=T12345678
SLACK_LISTEN_CHANNELS=#general,#random

# MCP Agent Configuration (if needed)
OPENAI_API_KEY=your-openai-api-key
MODEL_NAME=gpt-4
```

### 3. Install Dependencies

```bash
pip install slack-bolt python-dotenv certifi
# Install any additional dependencies for your MCP agent
```

## Running the Bot

### Standalone Mode

Run the integration directly:

```bash
python slack_agent_integration.py
```

### Integrated Mode

To use the bot as part of a larger application, import and use the `SlackAgentIntegration` class:

```python
from slack_agent_integration import SlackAgentIntegration

async def start_bot():
    integration = SlackAgentIntegration()
    await integration.initialize()
    await integration.start()
    
    # Keep your application running
    # ...
    
    # When shutting down:
    await integration.stop()
```

## Architecture

The project consists of two main components:

1. **SlackBotListener** (`slackbot_listener.py`): Handles the Slack connection and message events
2. **SlackAgentIntegration** (`slack_agent_integration.py`): Connects the bot to the MCP agent system

The flow is:
1. Slack sends message events to the SlackBotListener
2. The listener forwards messages to the integration
3. The integration formats and sends messages to the MCP agent
4. The agent uses the slack-mcp-server to respond back to Slack

## Customization

### Listening to Different Channels

Update the `SLACK_LISTEN_CHANNELS` environment variable with a comma-separated list of:
- Channel IDs (e.g., `C12345678`)
- Channel names with # prefix (e.g., `#general`)

### Modifying Agent Behavior

Edit the system prompt in the `_create_agent` method in `slack_agent_integration.py` to change how the agent responds to messages.

## Troubleshooting
### Bot Not Responding

1. Check the logs for error messages
2. Verify the bot has been invited to the channels it's supposed to monitor
3. Ensure all required scopes are enabled in the Slack app settings

### Connection Issues

1. Verify your tokens are correct
2. Make sure the `SLACK_APP_TOKEN` has the `connections:write` scope
3. Check if your firewall is blocking WebSocket connections

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.