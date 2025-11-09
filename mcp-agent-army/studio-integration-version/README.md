# MCP Agent Army - Studio Integration Version

This is the LiveAgent Studio integration version of the MCP Agent Army - a powerful multi-agent system that leverages the Model Context Protocol (MCP) to orchestrate specialized AI agents to perform various tasks through third-party services.

## Features

- **Multi-agent Architecture**: Distribute tasks to specialized agents for better handling of complex operations
- **FastAPI Endpoint**: Easy integration with LiveAgent Studio or other frameworks
- **Dynamic Model Selection**: Automatically choose the optimal OpenAI model based on task complexity
- **Supabase Integration**: Store conversation history in a database
- **Asynchronous Processing**: Efficiently manage multiple operations in parallel

## API Endpoints

### POST /api/mcp-agent-army

Send a request to the MCP Agent Army.

**Request Body**:
```json
{
  "query": "Your question or instruction",
  "user_id": "unique-user-id",
  "request_id": "unique-request-id",
  "session_id": "unique-session-id",
  "auto_model_selection": true
}
```

- `query`: The user's question or instruction
- `user_id`: Unique identifier for the user
- `request_id`: Unique identifier for this specific request
- `session_id`: Session identifier to maintain conversation history
- `auto_model_selection`: (Optional) Enable/disable dynamic model selection (default: false)

**Response**:
```json
{
  "success": true,
  "model_used": "gpt-4o-mini",
  "rephrased_query": "Optimized version of the query" 
}
```

### POST /api/toggle-model-selection

Toggle the auto model selection feature for a session.

**Request Body**:
```json
{
  "session_id": "unique-session-id",
  "enable": true
}
```

**Response**:
```json
{
  "success": true,
  "auto_model_selection": true
}
```

## Dynamic Model Selection

The dynamic model selection feature automatically chooses the most appropriate OpenAI model for each user request:

- **gpt-3.5-turbo**: For simple queries, factual questions, and basic tasks
- **gpt-4o-mini**: For medium complexity tasks requiring good performance at reasonable cost
- **gpt-4o**: For complex reasoning, creative tasks, and detailed analysis

When enabled, a specialized model selection agent:
1. Analyzes the user's query
2. Selects the most suitable model
3. Rephrases the query to be optimized for the selected model
4. Processes the request with the chosen model

The feature can be enabled/disabled per request or toggled for an entire session.

## Deployment

### Docker

Build and run using Docker:

```bash
# Build the Docker image
docker build -t mcp-agent-army .

# Run the container
docker run -p 8001:8001 --env-file .env mcp-agent-army
```

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI app:
   ```bash
   python mcp_agent_army_endpoint.py
   ```

## Environment Variables

Create an `.env` file with these variables:

- `PROVIDER`: Your LLM provider (OpenAI, OpenRouter, Ollama)
- `BASE_URL`: API base URL for your LLM provider  
- `LLM_API_KEY`: Your LLM API key
- `MODEL_CHOICE`: Default model (e.g., gpt-4o-mini)
- `API_BEARER_TOKEN`: Bearer token for authenticating API requests
- `SUPABASE_URL`: URL for your Supabase instance
- `SUPABASE_SERVICE_KEY`: Supabase service key
- Various API keys for the MCP servers (see main README) 