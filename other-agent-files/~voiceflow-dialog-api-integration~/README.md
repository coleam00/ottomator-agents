# Voiceflow Dialog API Integration

Author: [Cole Medin](https://www.youtube.com/@ColeMedin)

> For detailed information about the Voiceflow Dialog API and how to best utilize it with your Voiceflow agents, refer to the [official Voiceflow Dialog API documentation](https://docs.voiceflow.com/reference/overview).

This integration provides seamless connectivity between Voiceflow agents and the Live Agent Studio. The key feature of this integration is its zero-configuration approach - **once you build an agent in Voiceflow, it's immediately ready to use in the Live Agent Studio without any additional setup or customization**.

This API endpoint demonstrates the implementation details of how Voiceflow agents are integrated with the Live Agent Studio. While you don't need to modify this code to use your Voiceflow agents, understanding how it works can be valuable for advanced use cases.

## Overview

This integration enables:
- Instant deployment of Voiceflow agents to Live Agent Studio
- Real-time conversation handling through Voiceflow's Dialog API
- Automatic message history tracking
- Support for rich responses (text, buttons, carousels - future enhancement)
- Secure API authentication
- Session management for maintaining conversation context

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Supabase account (for conversation storage)
- Voiceflow account with an API key
- Basic understanding of:
  - FastAPI and async Python
  - RESTful APIs
  - Voiceflow Dialog API

## Core Components

### 1. FastAPI Application (`voiceflow_integration.py`)

The integration is built using FastAPI, providing:

- **Voiceflow Integration**
  - Dialog API communication
  - Session state management
  - Rich response handling

- **Authentication**
  - Bearer token validation
  - Secure endpoint protection
  - API key management

- **Database Integration**
  - Supabase connection for message storage
  - Conversation history tracking
  - Session management

### 2. Data Models

#### Request Model
```python
class AgentRequest(BaseModel):
    query: str        # The user's input text or JSON action
    user_id: str      # Unique identifier for the user
    request_id: str   # Unique identifier for this request
    session_id: str   # Current conversation session ID
```

#### Response Model
```python
class AgentResponse(BaseModel):
    success: bool     # Indicates if the request was processed successfully
```

### 3. Database Schema

The integration uses Supabase tables with the following structure:

#### Messages Table
```sql
messages (
    id: uuid primary key
    created_at: timestamp with time zone
    session_id: text
    message: jsonb {
        type: string       # 'human' or 'assistant'
        content: string    # The message content
        data: jsonb       # Voiceflow response data
    }
)
```

## Frontend Integration

The `VoiceflowFrontendComponent.tsx` demonstrates how Voiceflow traces from the Dialog API are integrated with the Live Agent Studio frontend. This component serves as an example of creating custom frontend components to display rich responses from your agent using the `data` field of the AI response instead of raw text output.

### Key Features
- Handles various Voiceflow trace types (text, choice, knowledgeBase)
- Renders interactive buttons for choice responses
- Processes structured message data from Voiceflow's Dialog API
- Maintains conversation context through session management

### Example Usage
```typescript
interface VoiceflowTrace {
  type: string;
  payload: {
    message?: string;
    slate?: {
      content: Array<{
        children: Array<{
          text: string;
        }>;
      }>;
    };
    buttons?: VoiceflowButton[];
  };
}

// Component renders different UI elements based on trace type
const renderTrace = (trace: VoiceflowTrace) => {
  switch (trace.type) {
    case 'text':
      return <p>{trace.payload.message}</p>;
    case 'choice':
      return <ButtonGroup buttons={trace.payload.buttons} />;
    // ... handle other trace types
  }
};
```

## Setup

1. **Environment Variables**
   Create a `.env` file with:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_key
   API_BEARER_TOKEN=your_api_token
   VOICEFLOW_AGENT_API_KEY=your_voiceflow_api_key
   ```

2. **Local Development**
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Run the server
   python voiceflow_integration.py
   ```

3. **Docker Deployment**
   ```bash
   # Build the container
   docker build -t voiceflow-integration .

   # Run the container
   docker run -d --name voiceflow-integration -p 8001:8001 --env-file .env voiceflow-integration
   ```

## Using Your Voiceflow Agent

1. Create and publish your agent in Voiceflow
2. Copy your Dialog API key from Voiceflow
3. Add the API key to your environment variables
4. Your agent is now ready to use in the Live Agent Studio!

## API Endpoints

### POST /api/sample-voiceflow-agent
Handles all communication between Live Agent Studio and your Voiceflow agent.

**Request:**
```json
{
    "query": "Hello!",
    "user_id": "user123",
    "request_id": "req123",
    "session_id": "sess123"
}
```

**Response:**
```json
{
    "success": true
}
```

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
