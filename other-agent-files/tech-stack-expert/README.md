# Tech Stack Expert Agent

Author: [Cole Medin](https://www.youtube.com/@ColeMedin)

A specialized n8n agent that helps users determine the ideal tech stack for their full-stack applications. This agent demonstrates how to create a basic conversational AI assistant in n8n.

## Features

- Conducts guided conversations to understand project requirements
- Considers user experience with different technologies
- Provides tailored recommendations for:
  - Frontend frameworks
  - Backend technologies
  - Authentication solutions
  - Database systems
  - LLM integration
- Maintains conversation context using Postgres chat memory

## How It Works

The agent follows a structured conversation flow:

1. Gathers information about the application concept
2. Identifies which AI coding assistant the user is working with
3. Assesses user's experience with frontend technologies
4. Evaluates backend technology experience
5. Determines expected user scale
6. Considers specific technology requirements

## Tech Stack Recommendations

The agent provides recommendations based on specific criteria:

### Frontend
- Streamlit for simple, low-traffic applications
- Next.js or React/Vite for complex applications
- Adapts to AI coding assistant constraints (e.g., React/Vite for Bolt.new users)

### Backend
- n8n/Flowise for simple AI agents
- LangChain + LangGraph for complex AI applications
- FastAPI (Python) or Express (JavaScript) for APIs
- Go for high-performance, non-AI applications

### Authentication
- Supabase Auth for standard cases
- Auth0 for complex SSO requirements

### Database
- Supabase (PostgreSQL) with PGVector for RAG applications
- MongoDB/Firebase for specific use cases

### LLM Integration
- Claude 3.5 (Sonnet/Haiku) for general use
- Ollama with Qwen models for private data
- Llama 3.2 for vision capabilities

## Implementation Details

This agent is built using the Live Agent Studio framework and demonstrates:

1. **Webhook Integration**
   - Handles incoming requests
   - Processes user queries securely

2. **Conversation Management**
   - Uses Postgres for chat memory
   - Maintains context across multiple interactions

3. **AI Model Integration**
   - Leverages Claude 3.5 Haiku
   - Structured prompting for consistent responses

4. **State Management**
   - Tracks conversation progress
   - Maintains user session context

## Usage

1. Send a POST request to the webhook endpoint with:
```json
{
    "query": "Your question about tech stack",
    "user_id": "unique-user-identifier",
    "request_id": "request-tracking-id",
    "session_id": "conversation-session-id"
}
```

2. The agent will guide you through a series of questions to understand your needs

3. After gathering necessary information, it provides a comprehensive tech stack recommendation

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
