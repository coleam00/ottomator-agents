# MCP Agent Army

A powerful multi-agent system built with [Archon](https://github.com/coleam00/Archon), the AI agent that builds other AI Agents. This system leverages the Model Context Protocol (MCP) and Pydantic AI to orchestrate specialized AI agents to perform various tasks through third-party services (Slack, Firecrawl, Airtable, etc.).

Specialized agents are important because LLMs get overwhelmed very easily if you give a single agent too many tools.
Splitting the tools for each service into subagents is the best way to give an agent system many capabilities while
still keeping each individual prompt relatively short.

See `prompt.txt` for the initial prompt I gave to Archon to build this AI agent army. Keep in mind I had to iterate on the agent for a few prompts with Archon to get it right, but it still gave a fantastic starting point after one shot!

## Overview

This system uses a primary orchestration agent that delegates tasks to specialized subagents, each with expertise in a specific third-party service:

- **Airtable Agent**: Manages Airtable databases and records
- **Brave Search Agent**: Performs web searches and retrieves information
- **Filesystem Agent**: Handles file operations and directory management
- **GitHub Agent**: Interacts with GitHub repositories, issues, and PRs
- **Slack Agent**: Sends messages and manages Slack communications
- **Firecrawl Agent**: Extracts data from websites through web crawling

## Key Features

- **Multi-agent Architecture**: Distribute tasks to specialized agents for better handling of complex operations
- **Dynamic Model Selection**: Automatically choose the optimal OpenAI model based on the complexity of the user's request (NEW!)
   to use it type "enable auto model" to chat. You will be prompted with "Enabled auto model selection"
- **Integrated MCP Servers**: Connect with various third-party services through standardized interfaces
- **Asynchronous Processing**: Efficiently manage multiple operations in parallel

## Requirements

- Python 3.9+
- Node.js and npm (for MCP servers)
- API keys for various services (see `.env.example`)

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
4. Copy `.env.example` to `.env` and fill in your API keys and configuration

## Environment Variables

Set the following environment variables in your `.env` file:

- `PROVIDER`: Your LLM provider (OpenAI, OpenRouter, Ollama)
- `BASE_URL`: API base URL for your LLM provider
- `LLM_API_KEY`: Your LLM API key
- `MODEL_CHOICE`: The default model to use (e.g., gpt-4o-mini)
- `BRAVE_API_KEY`: API key for Brave Search
- `AIRTABLE_API_KEY`: API key for Airtable
- `GITHUB_TOKEN`: Personal access token for GitHub
- `SLACK_BOT_TOKEN`: Bot token for Slack
- `SLACK_APP_TOKEN`: App token for Slack
- `FIRECRAWL_API_KEY`: API key for Firecrawl

## Usage

Run the main script:

```bash
python mcp_agent_army.py
```

Enter your requests at the prompt. The primary agent will analyze your request and delegate it to the appropriate specialized agent.

### Dynamic Model Selection

To enable the automatic model selection feature:
```
enable auto model
```

To disable it and revert to using the default model:
```
disable auto model
```

When enabled, the system will:
1. Analyze your request to determine the best OpenAI model (gpt-3.5-turbo, gpt-4o-mini, or gpt-4o)
2. Optimize your query for the selected model
3. Process your request with the chosen model

Example requests:
- "Search for the latest AI research papers on multi-agent systems"
- "Create a new file called test.txt with 'Hello World' content"
- "Check the status of my GitHub repository issues"
- "Send a message to the #general channel in Slack"
- "Extract product information from the Amazon page for iPhone 15"
- "Create a new record in my Airtable database"

## Architecture

The system uses AsyncExitStack to manage all MCP servers in a single context, making it efficient and robust. Each subagent is initialized with its own MCP server and system prompt that defines its expertise.

The primary agent has tools to invoke each subagent, allowing it to delegate tasks based on the user's request.

## License

MIT
