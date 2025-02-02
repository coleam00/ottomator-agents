# File Processing Agent for Live Agent Studio

Author: [Cole Medin](https://www.youtube.com/@ColeMedin)

This is a specialized Python FastAPI agent that demonstrates how to handle file uploads in the Live Agent Studio. It shows how to process, store, and leverage file content in conversations with AI models.

This agent builds upon the foundation laid out in [`~sample-python-agent~/sample_supabase_agent.py`](../~sample-python-agent~/sample_supabase_agent.py), extending it with file handling capabilities.

Not all agents need file handling which is why the sample Python agent is kept simple and this one is available to help you build agents with file handling capabilities. The Live Agent Studio has file uploading built in and the files are sent in the exact format shown in this agent.

## Overview

This agent extends the base Python agent template to showcase file handling capabilities:
- Process uploaded files in base64 format
- Store file content with conversation history
- Integrate file content into AI model context
- Maintain conversation continuity with file references
- Handle multiple files in a single conversation

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Supabase account (for conversation storage)
- OpenAI API key
- Basic understanding of:
  - FastAPI and async Python
  - Base64 encoding/decoding
  - OpenAI API
  - Supabase

## Core Components

### 1. File Processing

The agent includes robust file handling:
- Base64 decoding of uploaded files
- Text extraction and formatting
- Persistent storage of file data
- Context integration for AI responses

### 2. Conversation Management

Built on the sample Supabase agent template, this agent adds:
- File metadata storage with messages
- File content integration in conversation history
- Contextual file reference handling

### 3. AI Integration

Seamless integration with OpenAI's GPT models:
- File content as conversation context
- Maintained context across multiple messages
- Intelligent responses based on file content

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_key
   API_BEARER_TOKEN=your_bearer_token
   OPENAI_API_KEY=your_openai_key
   ```

## Running the Agent

Start the agent with:
```bash
python file_agent.py
```

The agent will be available at `http://localhost:8001`.

## API Usage

Send requests to `/api/file-agent` with:
- `query`: Your question or prompt
- `files`: Array of file objects with:
  - `name`: Filename
  - `type`: MIME type
  - `base64`: Base64-encoded file content

Example request:
```json
{
  "query": "What does this file contain?",
  "files": [{
    "name": "example.txt",
    "type": "text/plain",
    "base64": "VGhpcyBpcyBhIHRlc3QgZmlsZS4="
  }],
  "session_id": "unique-session-id",
  "user_id": "user-id",
  "request_id": "request-id"
}
```

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
