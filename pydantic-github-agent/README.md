# Pydantic AI: GitHub Repository Analysis Agent

An intelligent GitHub repository analysis agent built using Pydantic AI, capable of analyzing GitHub repositories to answer user questions. The agent can fetch repository information, explore directory structures, and analyze file contents using the GitHub API.

## Features

- Repository information retrieval (size, description, etc.)
- Directory structure analysis
- File content examination
- Support for both OpenAI and OpenRouter models
- Command-line interface for interactive repository analysis

## Prerequisites

- Python 3.11+
- GitHub Personal Access Token (for private repositories)
- OpenRouter API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alimoradi296/ottomator-agents/tree/feature/enhanced-github-agent)
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
   OPENAI_API_KEY=your_openrouter_api_key
   OPEN_AI_BASE_URL=your_base_url_openai
   LLM_MODEL=your_chosen_model  # e.g., deepseek/deepseek-chat
   ```

## Usage

### Command Line Interface

Run the command-line interface to interact with the GitHub analysis agent:

```bash
python github_agent_ai.py
```

Example queries you can ask:
- "What's the structure of repository https://github.com/username/repo?"
- "Show me the contents of the main Python file in https://github.com/username/repo"
- "What are the key features of repository https://github.com/username/repo?"

## Configuration

### LLM Models

You can configure different LLM models by setting the `LLM_MODEL` environment variable. The agent uses OpenRouter as the API endpoint, supporting various models:

```env
LLM_MODEL=deepseek/deepseek-chat  # Default model
```

### API Keys

- **GitHub Token**: Generate a Personal Access Token from [GitHub Settings](https://github.com/settings/tokens)
- ** API Key**: Get your API key from [OpenRouter](https://openrouter.ai/)

## Project Structure

- `github_agent_ai.py`: Core agent implementation with GitHub API integration
- `requirements.txt`: Project dependencies

## Error Handling

The agent includes built-in retries for API calls and proper error handling for:
- Invalid GitHub URLs
- Rate limiting
- Authentication issues
- File not found errors

