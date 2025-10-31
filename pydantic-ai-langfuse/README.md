# Pydantic AI with Langfuse Observability

A powerful multi-agent system built with Pydantic AI and integrated with Langfuse for observability. This project demonstrates how to create a system of specialized AI agents that can perform various tasks through third-party services while monitoring their performance and behavior through Langfuse. You can also use LLMs from OpenAI, OpenRouter, or Ollama!

## Overview

This project showcases the integration of Pydantic AI with Langfuse for agent observability. It includes:

1. **Main Agent (`pydantic_ai_langfuse_agent.py`)**: A primary orchestration agent that delegates tasks to specialized subagents
   - Tracks session IDs and user IDs for the Langfuse traces
   - Can use any OpenAI, OpenRouter, or Ollama LLM (see `.env.example` for instructions)
   - Uses MCP servers for all the agent tools
2. **Iteration Examples**:
   - `simple_langfuse.py`: Basic example of Langfuse integration with OpenAI
   - `simple_pydantic_ai.py`: Simple Pydantic AI agent with Brave Search capability and Langfuse integration

The main agent system uses a primary orchestration agent that delegates tasks to specialized subagents, each with expertise in a specific third-party service:

- **Airtable Agent**: Manages Airtable databases and records
- **Brave Search Agent**: Performs web searches and retrieves information
- **Filesystem Agent**: Handles file operations and directory management
- **GitHub Agent**: Interacts with GitHub repositories, issues, and PRs
- **Slack Agent**: Sends messages and manages Slack communications
- **Firecrawl Agent**: Extracts data from websites through web crawling

## Requirements

- Python 3.11+
- Node.js and npm (for MCP servers)
- Langfuse (either managed in the cloud or self-hosted)
- API keys for various services and Langfuse

## Installation

1. Clone this repository
2. Set up a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on the environment variables listed below

Note: a `.env.example` file is already included in this folder. Copy it to `.env` and adjust values.

To run securely and fully-local (no MCP servers), set:

```
USE_MCP_SERVERS=false
```

## Langfuse Setup

Langfuse provides observability for your AI agents. You can either use Langfuse Cloud (managed service) or self-host Langfuse.

### Option 1: Langfuse Cloud (Managed)

1. Sign up at [Langfuse.com](https://langfuse.com)
2. Create a new project
3. Get your Public Key and Secret Key from the project settings
4. Add these keys to your `.env` file:
   ```
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

### Option 2: Self-hosted Langfuse

You have two options for self-hosting Langfuse:

1. **Use the local-ai-packaged repository**:
   - Clone [local-ai-packaged](https://github.com/coleam00/local-ai-packaged)
   - Follow the setup instructions to run Langfuse locally

2. **Follow the official Langfuse self-hosting guide**:
   - Visit the [Langfuse self-hosting documentation](https://langfuse.com/self-hosting)
   - Choose from deployment options including Docker, Kubernetes, or VM

After setting up your self-hosted instance, configure your environment variables:

```
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=http://localhost:3002  # or your custom URL
```

## Environment Variables

Create a `.env` file with the following variables:

```
# LLM Provider
PROVIDER=openai (can be ollama or openrouter too, see .env.example for instructions)
BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_openai_api_key
MODEL_CHOICE=gpt-4.1-mini

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=http://localhost:3002  # or https://cloud.langfuse.com for managed service

# MCP Service API Keys
BRAVE_API_KEY=your_brave_api_key
AIRTABLE_API_KEY=your_airtable_api_key
GITHUB_TOKEN=your_github_token
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_TEAM_ID=your_slack_team_id
FIRECRAWL_API_KEY=your_firecrawl_api_key
LOCAL_FILE_DIR=your_local_file_directory
```

## Usage

### Running the Main Agent

The main agent orchestrates all subagents and provides a comprehensive interface:

```bash
python pydantic_ai_langfuse_agent.py
```

Enter your requests at the prompt. The primary agent will analyze your request and delegate it to the appropriate specialized agent.

### Running the Simple Examples

For a basic demonstration of Langfuse integration:

```bash
cd iterations
python simple_langfuse.py
```

For a simple Pydantic AI agent with Brave Search:

```bash
cd iterations
python simple_pydantic_ai.py
```

Be sure to set the .env file specifically in the `iterations/` folder if using this simpler examples!

## Viewing Traces in Langfuse

After running your agents, you can view the traces in Langfuse:

1. Open your Langfuse dashboard:
   - Managed: https://cloud.langfuse.com
   - Self-hosted: http://localhost:3002 (or your custom URL)
2. Navigate to the Traces section
3. View detailed information about your agent runs, including:
   - Execution time
   - Model calls
   - Input/output data
   - Custom metrics and events

## Architecture

The system uses AsyncExitStack to manage all MCP servers in a single context, making it efficient and robust. Each subagent is initialized with its own MCP server and system prompt that defines its expertise.

The primary agent has tools to invoke each subagent, allowing it to delegate tasks based on the user's request.

Langfuse integration provides observability through:
- Tracing of all agent interactions
- Monitoring of model calls and tool calls
- Performance analytics (latency, tokens, cost)
- Tracking conversations for individual users

## License

MIT
