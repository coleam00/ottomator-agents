# AI Agent Fundamentals

A minimal demonstration of building an AI agent with Pydantic AI.

## Features

- Structured system prompt with clear sections
- Tool usage (add_numbers function)
- Conversation history preservation
- OpenRouter integration for model access
- Langfuse observability for production monitoring (Optional)

## Prerequisites

- Python 3.10+
- UV package manager
- OpenRouter API key ([get one here](https://openrouter.ai/))
- Optional - Langfuse account ([sign up here](https://cloud.langfuse.com/))

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment variables:
```bash
cp .env.example .env
```

3. Edit `.env` and add your credentials:
```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key
MODEL=anthropic/claude-haiku-4.5

# Optional for agent observability
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

## Usage

Run the agent:
```bash
uv run python agent.py
```

Chat with the agent and try asking it to add numbers. Type `exit` to quit.

All conversations are automatically traced in Langfuse for monitoring and debugging. View your traces at [cloud.langfuse.com](https://us.cloud.langfuse.com).

## Project Structure

- `agent.py` - Main agent implementation with CLI
- `prompt.py` - Structured system prompt
- `observability.py` - Langfuse observability setup
- `.env.example` - Environment variable template
- `pyproject.toml` - Project dependencies

## How It Works

The agent uses:
- **Pydantic AI** for the agent framework
- **OpenRouter** via OpenAIChatModel provider for model access
- **Tools** to extend agent capabilities (e.g., add_numbers)
- **Message history** to maintain conversation context
- **Langfuse** for observability and tracing via OpenTelemetry
