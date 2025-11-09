# Archon API Reference

Complete reference for all Archon REST API endpoints. This mirrors the functionality of the 14 MCP tools.

## Base Configuration

**API Server:** `http://localhost:8181` (configurable via `ARCHON_SERVER_PORT`)

**Standard Headers:**
```
Content-Type: application/json
Accept: application/json
```

## Knowledge Management Endpoints

### Search Knowledge Base

**Endpoint:** `POST /api/knowledge-items/search`

**Purpose:** Semantic search across knowledge base using RAG with hybrid search, reranking, and contextual embeddings.

**Request Body:**
```json
{
  "query": "How to implement authentication",
  "top_k": 10,
  "use_reranking": true,
  "search_strategy": "hybrid",
  "filters": {
    "source_type": ["documentation"],
    "tags": ["auth"]
  }
}
```

**Parameters:**
- `query` (required): Search query text
- `top_k` (optional, default: 10): Number of results to return
- `use_reranking` (optional, default: true): Apply cross-encoder reranking for better results
- `search_strategy` (optional, default: "hybrid"): Search method
  - `"hybrid"`: Combines semantic and keyword search (recommended)
  - `"semantic"`: Pure vector similarity search
  - `"keyword"`: Traditional keyword-based search
- `filters` (optional): Filter results
  - `source_type`: Array of types (e.g., ["documentation", "code"])
  - `tags`: Array of tags to match

**Response:**
```json
{
  "results": [
    {
      "id": "chunk-uuid",
      "content": "Authentication can be implemented using...",
      "score": 0.92,
      "metadata": {
        "source_url": "https://docs.example.com/auth",
        "source_type": "documentation",
        "code_example": true,
        "tags": ["auth", "security"]
      }
    }
  ],
  "search_metadata": {
    "strategy": "hybrid",
    "reranking_applied": true,
    "total_results": 50
  }
}
```

### List Knowledge Items

**Endpoint:** `GET /api/knowledge-items`

**Purpose:** List all indexed content in knowledge base.

**Query Parameters:**
- `limit` (optional, default: 50): Maximum results
- `offset` (optional, default: 0): Pagination offset
- `source_type` (optional): Filter by type (documentation, code, pdf, etc.)

**Example:** `GET /api/knowledge-items?limit=20&source_type=documentation`

**Response:**
```json
{
  "items": [
    {
      "id": "item-uuid",
      "title": "Authentication Guide",
      "source_url": "https://docs.example.com/auth",
      "source_type": "documentation",
      "chunks_count": 15,
      "created_at": "2025-10-21T10:00:00Z"
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

### Crawl Website

**Endpoint:** `POST /api/knowledge-items/crawl`

**Purpose:** Crawl website and add to knowledge base with automatic sitemap detection and code example extraction.

**Request Body:**
```json
{
  "url": "https://docs.example.com",
  "crawl_depth": 3,
  "follow_links": true,
  "sitemap_url": null
}
```

**Parameters:**
- `url` (required): Starting URL to crawl
- `crawl_depth` (optional, default: 3, max: 5): Depth of recursive crawling
- `follow_links` (optional, default: true): Whether to follow links in pages
- `sitemap_url` (optional): Direct URL to sitemap if known

**Features:**
- Automatically detects sitemaps and llms.txt
- Extracts code examples for enhanced search
- Supports recursive crawling with configurable depth
- Real-time progress via WebSocket

**Response:**
```json
{
  "success": true,
  "crawl_id": "uuid-here",
  "pages_queued": 150,
  "message": "Crawling started"
}
```

**WebSocket Progress:** Connect to Socket.IO at base URL to receive real-time updates:
```javascript
socket.on('crawl:progress', (data) => {
  // { crawl_id, pages_crawled, total_pages, status }
});
```

### Upload Document

**Endpoint:** `POST /api/knowledge-items/upload`

**Purpose:** Upload documents (PDF, Word, Markdown, text) with intelligent chunking.

**Request:** Multipart form data
```
file: [binary file data]
metadata: {
  "source_type": "pdf",
  "tags": ["api-docs", "reference"]
}
```

**Supported Formats:**
- PDF (.pdf)
- Word documents (.docx, .doc)
- Markdown (.md)
- Text files (.txt)

**Response:**
```json
{
  "success": true,
  "document_id": "uuid-here",
  "chunks_created": 25,
  "processing_status": "completed"
}
```

### Get RAG Sources

**Endpoint:** `GET /api/rag/sources`

**Purpose:** Get list of all available knowledge sources in the RAG system.

**⚠️ LIMITATION:** This endpoint returns LIMITED metadata. For full metadata including word count, code examples count, and estimated pages, use `GET /api/knowledge-items` instead.

**Response:**
```json
{
  "sources": [
    {
      "source_id": "source-uuid",
      "title": "Next.js Documentation",
      "summary": "Documentation for Next.js framework...",
      "metadata": {
        "source_type": "documentation"
      },
      "total_words": 245000,
      "created_at": "2025-10-15T10:00:00Z",
      "updated_at": "2025-10-21T10:00:00Z"
    }
  ],
  "count": 15
}
```

**Missing from this endpoint:** `word_count`, `code_examples_count`, `estimated_pages`, detailed `metadata` object.

**Recommended:** Use `GET /api/knowledge-items` for complete metadata (see above).

### Get Database Metrics

**Endpoint:** `GET /api/database/metrics`

**Purpose:** Get database metrics including total documents, chunks, code examples, and other statistics.

**Response:**
```json
{
  "total_documents": 150,
  "total_chunks": 3450,
  "total_code_examples": 287,
  "sources_by_type": {
    "documentation": 85,
    "pdf": 35,
    "code": 20,
    "markdown": 10
  },
  "total_projects": 12,
  "total_tasks": 145,
  "storage_size_mb": 512.5,
  "last_updated": "2025-10-21T10:30:00Z"
}
```

### ⚠️ Deprecated Endpoints

**Endpoint:** `GET /api/knowledge-items/sources` (DEPRECATED)

**Status:** This endpoint is deprecated and returns an empty array. Use `GET /api/rag/sources` instead for listing all available sources.

## Project Management Endpoints

### List Projects

**Endpoint:** `GET /api/projects`

**Purpose:** Get all projects with hierarchical structure.

**Response:**
```json
{
  "projects": [
    {
      "id": "project-uuid",
      "name": "API Redesign",
      "description": "Complete API overhaul",
      "features_count": 5,
      "tasks_count": 23,
      "created_at": "2025-10-01T10:00:00Z",
      "updated_at": "2025-10-21T10:00:00Z"
    }
  ]
}
```

### Get Project

**Endpoint:** `GET /api/projects/{project_id}`

**Purpose:** Get detailed project information including features and tasks.

**Response:**
```json
{
  "id": "project-uuid",
  "name": "API Redesign",
  "description": "Complete API overhaul",
  "features": [
    {
      "id": "feature-uuid",
      "name": "Authentication",
      "tasks_count": 8
    }
  ],
  "tasks": [
    {
      "id": "task-uuid",
      "title": "Implement OAuth2",
      "status": "in_progress",
      "feature_id": "feature-uuid"
    }
  ],
  "created_at": "2025-10-01T10:00:00Z"
}
```

### Create Project

**Endpoint:** `POST /api/projects`

**Purpose:** Create new project.

**Request Body:**
```json
{
  "name": "New Project",
  "description": "Project description"
}
```

**Response:**
```json
{
  "success": true,
  "project": {
    "id": "project-uuid",
    "name": "New Project",
    "description": "Project description",
    "created_at": "2025-10-21T10:00:00Z"
  }
}
```

## Task Management Endpoints

### Create Task

**Endpoint:** `POST /api/tasks`

**Purpose:** Create task within a project.

**Request Body:**
```json
{
  "project_id": "project-uuid",
  "title": "Implement login endpoint",
  "description": "Create /auth/login endpoint with JWT",
  "status": "todo",
  "feature_id": "feature-uuid"
}
```

**Parameters:**
- `project_id` (required): UUID of the parent project
- `title` (required): Task title
- `description` (optional): Detailed description
- `status` (optional, default: "todo"): Task status
  - `"todo"`: Not started
  - `"in_progress"`: Currently working
  - `"done"`: Completed
  - `"blocked"`: Blocked by dependencies
- `feature_id` (optional): UUID of parent feature for hierarchical organization

**Response:**
```json
{
  "success": true,
  "task": {
    "id": "task-uuid",
    "project_id": "project-uuid",
    "title": "Implement login endpoint",
    "status": "todo",
    "created_at": "2025-10-21T10:00:00Z"
  }
}
```

### Update Task

**Endpoint:** `PUT /api/tasks/{task_id}`

**Purpose:** Update task status, title, description, or other fields.

**Request Body:**
```json
{
  "status": "in_progress",
  "description": "Updated description",
  "title": "New title"
}
```

**Response:**
```json
{
  "success": true,
  "task": {
    "id": "task-uuid",
    "status": "in_progress",
    "updated_at": "2025-10-21T10:30:00Z"
  }
}
```

### List Tasks

**Endpoint:** `GET /api/tasks`

**Purpose:** List tasks with optional filtering.

**Query Parameters:**
- `project_id` (optional): Filter by project UUID
- `feature_id` (optional): Filter by feature UUID
- `status` (optional): Filter by status (todo, in_progress, done, blocked)
- `limit` (optional, default: 50): Maximum results
- `offset` (optional, default: 0): Pagination offset

**Example:** `GET /api/tasks?project_id=xyz&status=in_progress&limit=20`

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-uuid",
      "project_id": "project-uuid",
      "title": "Implement OAuth2",
      "status": "in_progress",
      "description": "OAuth2 implementation details",
      "created_at": "2025-10-15T10:00:00Z",
      "updated_at": "2025-10-21T10:00:00Z"
    }
  ],
  "total": 8,
  "limit": 20,
  "offset": 0
}
```

### Get Task

**Endpoint:** `GET /api/tasks/{task_id}`

**Purpose:** Get detailed task information.

**Response:**
```json
{
  "id": "task-uuid",
  "project_id": "project-uuid",
  "feature_id": "feature-uuid",
  "title": "Implement OAuth2",
  "description": "Detailed task description",
  "status": "in_progress",
  "created_at": "2025-10-15T10:00:00Z",
  "updated_at": "2025-10-21T10:00:00Z"
}
```

## Document Management Endpoints

### List Documents

**Endpoint:** `GET /api/documents`

**Purpose:** Get project documents.

**Query Parameters:**
- `project_id` (optional): Filter by project UUID

**Example:** `GET /api/documents?project_id=xyz`

**Response:**
```json
{
  "documents": [
    {
      "id": "doc-uuid",
      "title": "API Specification",
      "project_id": "project-uuid",
      "version": 3,
      "created_at": "2025-10-10T10:00:00Z",
      "updated_at": "2025-10-21T10:00:00Z"
    }
  ]
}
```

### Create Document

**Endpoint:** `POST /api/documents`

**Purpose:** Create new document with optional project association.

**Request Body:**
```json
{
  "title": "API Specification",
  "content": "# API Spec\n\nComplete specification...",
  "project_id": "project-uuid"
}
```

**Parameters:**
- `title` (required): Document title
- `content` (required): Document content (supports Markdown)
- `project_id` (optional): Associate with project

**Response:**
```json
{
  "success": true,
  "document": {
    "id": "doc-uuid",
    "title": "API Specification",
    "version": 1,
    "created_at": "2025-10-21T10:00:00Z"
  }
}
```

### Update Document

**Endpoint:** `PUT /api/documents/{document_id}`

**Purpose:** Update document with automatic version tracking.

**Request Body:**
```json
{
  "title": "Updated API Specification",
  "content": "# Updated Spec\n\nNew content..."
}
```

**Response:**
```json
{
  "success": true,
  "document": {
    "id": "doc-uuid",
    "version": 4,
    "updated_at": "2025-10-21T10:30:00Z"
  }
}
```

## Error Responses

All endpoints return errors in standard format:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "url",
      "issue": "Invalid URL format"
    }
  },
  "timestamp": "2025-10-21T10:00:00Z"
}
```

**Common Error Codes:**
- `VALIDATION_ERROR`: Invalid request parameters
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Resource conflict (e.g., duplicate)
- `INTERNAL_ERROR`: Server error

## Configuration

**Environment Variables:**
```bash
# Core
ARCHON_SERVER_PORT=8181
HOST=localhost

# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key

# RAG
EMBEDDING_DIMENSIONS=1536  # For OpenAI text-embedding-3-small
OPENAI_API_KEY=configured-via-ui
```

**Interactive Documentation:**
- Swagger UI: `http://localhost:8181/docs`
- ReDoc: `http://localhost:8181/redoc`
- OpenAPI Spec: `http://localhost:8181/openapi.json`
