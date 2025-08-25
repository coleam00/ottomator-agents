# How to Connect and Interact with MaryPause AI Agent

The MaryPause AI Medical RAG Agent provides multiple ways to interact with it. Currently, the Render deployment is experiencing startup issues due to missing environment variables (database credentials). Here are your options:

## Option 1: Run Locally (Recommended for now)

### Quick Start
```bash
# 1. Set up environment variables
cp .env.example .env
# Edit .env with your database credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API server
python -m agent.api
# Server will run on http://localhost:8058
```

### Using the CLI Interface
```bash
# In another terminal, run the CLI
python cli.py

# Example interactions:
> What medical conditions are mentioned in the documents?
> Search for information about diabetes
> List available documents
```

## Option 2: Direct API Calls (When deployed or running locally)

### Available Endpoints

#### 1. Health Check
```bash
curl http://localhost:8058/health
```

#### 2. Chat with the Agent (Non-streaming)
```bash
curl -X POST http://localhost:8058/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What medical conditions are discussed in the documents?",
    "session_id": "optional-session-id"
  }'
```

#### 3. Chat with Streaming Responses
```bash
curl -X POST http://localhost:8058/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain diabetes treatment options",
    "session_id": "optional-session-id"
  }'
```

#### 4. Direct Search Operations

**Vector Search:**
```bash
curl -X POST http://localhost:8058/search/vector \
  -H "Content-Type: application/json" \
  -d '{
    "query": "diabetes symptoms",
    "top_k": 5
  }'
```

**Graph Search:**
```bash
curl -X POST http://localhost:8058/search/graph \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the relationships between diabetes and heart disease?",
    "max_results": 10
  }'
```

**Hybrid Search:**
```bash
curl -X POST http://localhost:8058/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "treatment options",
    "top_k": 5,
    "alpha": 0.7
  }'
```

#### 5. List Documents
```bash
curl http://localhost:8058/documents
```

#### 6. Get Session History
```bash
curl http://localhost:8058/sessions/{session_id}
```

## Option 3: Python Client Example

```python
import requests
import json

# Base URL - change to deployed URL when available
BASE_URL = "http://localhost:8058"

# Create a session
session_id = "user-123"

# Send a chat message
response = requests.post(
    f"{BASE_URL}/chat",
    json={
        "message": "What are the symptoms of diabetes?",
        "session_id": session_id
    }
)

result = response.json()
print("Agent Response:", result["response"])
print("Tools Used:", result["tools_used"])
print("Sources:", result["sources"])

# Stream responses for better UX
import sseclient

response = requests.post(
    f"{BASE_URL}/chat/stream",
    json={
        "message": "Explain the treatment options",
        "session_id": session_id
    },
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    if event.data:
        delta = json.loads(event.data)
        if delta["type"] == "content":
            print(delta["content"], end="", flush=True)
        elif delta["type"] == "tool_use":
            print(f"\n[Using tool: {delta['tool_name']}]")
```

## Option 4: Interactive API Documentation

When running locally, visit:
```
http://localhost:8058/docs
```

This provides an interactive Swagger UI where you can:
- Test all endpoints directly
- See request/response schemas
- Try different parameters

## Fixing the Render Deployment

The deployment needs environment variables configured in Render dashboard:

1. Go to: https://dashboard.render.com/web/srv-d2m587bipnbc738tbb8g/env
2. Add these required variables:
   - `DATABASE_URL` or `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
   - `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_CHOICE`
   - `EMBEDDING_PROVIDER`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`

3. After adding environment variables, trigger a new deployment

## Features of the Agent

The agent provides:
- **Vector Search**: Semantic similarity search across medical documents
- **Knowledge Graph Search**: Query relationships between medical entities
- **Hybrid Search**: Combines vector and text search for best results
- **Context-Aware Responses**: Maintains conversation history per session
- **Tool Transparency**: Shows which tools were used to generate responses
- **Source Attribution**: Provides sources for all information

## Common Use Cases

1. **Medical Information Retrieval**
   - "What are the symptoms of condition X?"
   - "List treatment options for Y"
   - "What medications are mentioned for Z?"

2. **Relationship Queries**
   - "How is diabetes related to heart disease?"
   - "What conditions are associated with obesity?"

3. **Document Exploration**
   - "What documents are available?"
   - "Summarize the content about vaccines"
   - "Find all mentions of specific medications"

## Troubleshooting

1. **502 Bad Gateway on Render**: The service needs environment variables configured
2. **Connection Refused Locally**: Make sure the API server is running (`python -m agent.api`)
3. **No Results**: Ensure you've ingested documents first (`python -m ingestion.ingest`)
4. **Neo4j Errors**: Check Neo4j is running and credentials are correct