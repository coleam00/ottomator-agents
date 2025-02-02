# Pydantic AI: GitHub Repository Analysis Agent

An intelligent GitHub repository analysis agent built using Pydantic AI, capable of analyzing GitHub repositories to answer user questions. The agent can fetch repository information, explore directory structures, and analyze file contents using the GitHub API.

## Features

- Repository information retrieval (size, description, etc.)
- Directory structure analysis
- File content examination
- Support for both OpenAI and OpenRouter models
- Available as both API endpoint and command-line interface

## Prerequisites

- Python 3.11+
- GitHub Personal Access Token (for private repositories)
- OpenRouter API key

## Installation and Usage with Python

1. Clone the repository:
```bash
git clone https://github.com/coleam00/ottomator-agents.git
cd ottomator-agents/pydantic-github-agent
```

2. Install dependencies (recommended to use a Python virtual environment):
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Rename `.env.example` to `.env`
   - Edit `.env` with your API keys and preferences:
   ```env
   GITHUB_TOKEN=your_github_token  # Required for private repos
   OPEN_ROUTER_API_KEY=your_openrouter_api_key
   LLM_MODEL=your_chosen_model  # e.g., deepseek/deepseek-chat
   SUPABASE_URL=your_supabase_url  # Only needed for endpoint
   SUPABASE_SERVICE_KEY=your_supabase_key  # Only needed for endpoint
   ```

### Running the FastAPI Endpoint

To run the agent as an API endpoint (also compatible with the oTTomator Live Agent Studio), run:

```bash
python github_agent_endpoint.py
```

The endpoint will be available at `http://localhost:8001`

### Command Line Interface

For a simpler interactive experience, you can use the command-line interface:

```bash
python cli.py
```

Example queries you can ask:
- "What's the structure of repository https://github.com/username/repo?"
- "Show me the contents of the main Python file in https://github.com/username/repo"
- "What are the key features of repository https://github.com/username/repo?"

## Installation and Usage with Docker

If you prefer using Docker, you don't need to install Python or any dependencies locally:

1. Clone the repository:
```bash
git clone https://github.com/coleam00/ottomator-agents.git
cd ottomator-agents/pydantic-github-agent
```

2. Set up environment variables:
   - Copy `.env.example` to `.env` and configure your API keys as shown in the Python installation section

3. Build and run with Docker:
```bash
docker build -t github-agent .
docker run -p 8001:8001 --env-file .env github-agent
```

The API endpoint will be available at `http://localhost:8001`

## Configuration

### LLM Models

You can configure different LLM models by setting the `LLM_MODEL` environment variable. The agent uses OpenRouter as the API endpoint, supporting various models:

```env
LLM_MODEL=deepseek/deepseek-chat  # Default model
```

### API Keys

- **GitHub Token**: Generate a Personal Access Token from [GitHub Settings](https://github.com/settings/tokens)
- **OpenRouter API Key**: Get your API key from [OpenRouter](https://openrouter.ai/)

## Project Structure

- `github_agent_ai.py`: Core agent implementation with GitHub API integration
- `cli.py`: Command-line interface for interacting with the agent
- `requirements.txt`: Project dependencies

## Live Agent Studio Version

If you're interested in seeing how this agent is implemented in the Live Agent Studio, check out the `studio-integration-version` directory. This contains the production version of the agent that runs on the platform.

## Error Handling

The agent includes built-in retries for API calls and proper error handling for:
- Invalid GitHub URLs
- Rate limiting
- Authentication issues
- File not found errors
