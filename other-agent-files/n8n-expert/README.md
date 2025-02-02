# n8n Expert Agent Workflow

Author: [Cole Medin](https://www.youtube.com/@ColeMedin)

The n8n Expert Agent helps you find and understand n8n automation workflows. Simply describe what you're trying to automate, and the agent will recommend relevant workflows to help you get started.

For example:
```
User: "I need to automatically post tweets when new blog posts are published"
Agent: Here are some recommended workflows to help you get started:
- WordPress to Twitter Auto-Poster
- Blog RSS Feed to Social Media
- Content Distribution Workflow
```

The agent analyzes your request and provides matching workflows from its knowledge base, along with explanations of how to implement and customize them for your needs.

This agent is in beta, especially as its knowledgebase grows! Workflows won't always be super related to your query and sometimes it won't have any. This agent will also involve into a conversational agent eventually to help you build n8n workflows!

## Features

- **Workflow Analysis**: Uses advanced LLM models to analyze n8n workflows and generate comprehensive summaries of:
  - Overall workflow purpose and functionality
  - Node configuration and connections
  - Potential variations and expansion suggestions

- **Legitimacy Validation**: Automatically detects and filters out test/spam workflows using AI-powered analysis

- **Workflow Processing**: 
  - Fetches workflow templates from n8n's public API
  - Converts workflows into HTML demo components
  - Generates vector embeddings for semantic search

- **Data Storage**: 
  - Stores processed workflows and analysis in Supabase
  - Maintains workflow metadata, summaries, and searchable embeddings

## Core Capabilities

1. **Intelligent Workflow Recommendations**
   - Processes natural language queries about automation needs
   - Uses LLM-powered analysis to match user requirements with existing workflows
   - Provides contextually relevant workflow suggestions based on user intent

2. **Semantic Search and Analysis**
   - Maintains a vector database of workflow descriptions and capabilities
   - Performs semantic similarity matching between user queries and workflow purposes
   - Filters and ranks workflows based on relevance to the user's needs

3. **Interactive Assistance**
   - Responds through a webhook-based API endpoint (`/invoke-n8n-expert`)
   - Maintains conversation context through session management
   - Provides detailed explanations of recommended workflows

### Use Cases

1. **Workflow Discovery**
   ```
   User: "I need to automatically post tweets when new blog posts are published"
   Agent: *Analyzes request and searches workflow database*
   Result: Returns relevant WordPress-to-Twitter automation workflows
   ```

2. **Implementation Guidance**
   - Explains how to adapt recommended workflows
   - Highlights key nodes and configurations
   - Suggests potential customizations

3. **Best Practices**
   - Recommends optimal node configurations
   - Suggests error handling and reliability improvements
   - Provides security and performance optimization tips

### Integration Features

1. **API Integration**
   - Secure webhook endpoint with header authentication
   - Structured JSON response format
   - Session-based conversation tracking

2. **Database Integration**
   - Supabase backend for workflow storage
   - Vector embeddings for semantic search
   - Message history and context management

3. **LLM Integration**
   - Advanced language model for query understanding
   - Workflow relevance scoring
   - Natural language response generation

### Example Interactions

1. **Workflow Search**
   ```
   Input: "How can I sync data between Airtable and Google Sheets?"
   Response: Provides relevant workflow examples with:
   - Step-by-step implementation guide
   - Required node configurations
   - Customization options
   ```

2. **Workflow Enhancement**
   ```
   Input: "How can I add error handling to my email automation?"
   Response: Suggests:
   - Error handling nodes to add
   - Retry configurations
   - Notification setup for failures
   ```

## Setup

1. Copy `.env.example` to `.env` and configure the following environment variables:
   - `LLM_MODEL`: Language model to use (default: 'gpt-4')
   - `EMBEDDING_MODEL`: Embedding model (default: 'text-embedding-3-small')
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_SERVICE_KEY`: Your Supabase service key

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Workflow Ingestion

Run the workflow ingestion script:
```bash
python ingest-n8n-workflows.py
```

This will:
1. Fetch workflows from n8n's template library
2. Validate each workflow
3. Generate analysis and embeddings
4. Store processed data in Supabase

### API Integration

The agent can be integrated into n8n workflows using the provided `N8N_Expert_Agent.json` workflow template, which handles:
- Webhook endpoints for workflow processing
- Query parameter handling
- User session management
- Database interactions

## Files

- `N8N_Expert_Agent.json`: Main n8n workflow template for the expert agent
- `ingest-n8n-workflows.py`: Python script for processing and storing n8n workflows
- `requirements.txt`: Python package dependencies
- `sql_script.sql`: Database schema and setup scripts
- `.env.example`: Example environment configuration file

## Dependencies

Key dependencies include:
- LangChain for LLM interactions
- OpenAI/Anthropic for language models
- Supabase for data storage
- Various Python utilities (see requirements.txt for full list)

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
