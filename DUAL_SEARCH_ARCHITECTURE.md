# Dual Search Architecture Documentation

## Overview

This document describes the dual search architecture implemented for the Medical RAG agent system. The system separates knowledge base queries from user interaction history, using different backend strategies optimized for each use case.

## Architecture Components

### 1. Knowledge Base Search (Direct Neo4j)
- **Purpose**: Query medical knowledge that was ingested directly into Neo4j
- **Technology**: Direct Neo4j Cypher queries via `neo4j_direct.py`
- **Data**: Medical documents, entities, relationships, facts
- **Scope**: Shared across all users (evidence-based medical information)

### 2. User Interaction History (Graphiti)
- **Purpose**: Store and retrieve conversation history and personal information
- **Technology**: Graphiti framework for temporal knowledge graphs
- **Data**: Conversation episodes, user-specific facts, personal health information
- **Scope**: Isolated per user using group_id partitioning

### 3. Vector Search (pgvector/Supabase)
- **Purpose**: Semantic similarity search across document chunks
- **Technology**: PostgreSQL with pgvector extension
- **Data**: Document embeddings and text chunks
- **Scope**: Shared medical literature and documents

## Implementation Details

### File Structure

```
agent/
├── neo4j_direct.py       # Direct Neo4j operations for knowledge base
├── graph_utils.py        # Graphiti operations for user history
├── tools.py              # Search tools with routing logic
├── agent.py              # Agent with registered tools
└── prompts.py            # System prompts with search strategy
```

### Key Classes and Functions

#### Neo4jDirectClient (`neo4j_direct.py`)
- `search_knowledge_base()`: Direct Cypher queries for medical facts
- `get_entity_relationships()`: Explore entity connections
- `find_paths_between_entities()`: Discover indirect relationships
- `get_graph_statistics()`: Knowledge base metrics

#### GraphitiClient (`graph_utils.py`)
- `add_conversation_episode()`: Store conversation turns
- `search()`: Query user history via Graphiti
- `get_session_episodes()`: Retrieve session-specific memories
- `add_fact_triples()`: Store user-specific facts

#### Search Tools (`tools.py`)
- `knowledge_base_search_tool()`: Direct Neo4j for medical knowledge
- `episodic_memory_search_tool()`: Graphiti for user history
- `graph_search_tool()`: Intelligent routing based on group_id
- `perform_comprehensive_search()`: Combine all search methods

## Search Routing Logic

The system intelligently routes queries based on context:

```python
# Routing decision in graph_search_tool
if group_ids is None or group_ids == ["0"] or "0" in group_ids:
    # Use direct Neo4j for shared knowledge base
    use_direct_neo4j = True
else:
    # Use Graphiti for user-specific data
    use_direct_neo4j = False
```

### When to Use Each Search Method

1. **Knowledge Base Search** (`knowledge_base_search`)
   - Medical facts and evidence
   - Symptoms and treatments
   - Medical entity relationships
   - Shared medical knowledge

2. **Episodic Memory** (`episodic_memory`)
   - Previous conversations
   - Personal health information
   - User-specific symptoms
   - Conversation continuity

3. **Vector Search** (`vector_search`)
   - Similar patient experiences
   - Detailed medical explanations
   - Document passages
   - Semantic similarity

4. **Hybrid Search** (`hybrid_search`)
   - Combining semantic and keyword matching
   - Comprehensive coverage
   - Medical terms with context

## Data Flow

### Knowledge Base Query Flow
```
User Query → Agent → knowledge_base_search → Neo4jDirectClient → Cypher Query → Results
```

### User History Query Flow
```
User Query → Agent → episodic_memory → GraphitiClient → Graphiti Search → Results
```

### Comprehensive Search Flow
```
User Query → Agent → perform_comprehensive_search → [
    → vector_search → pgvector
    → knowledge_base_search → Neo4j
    → episodic_memory → Graphiti
] → Combined Results
```

## Usage Examples

### Searching Medical Knowledge Base
```python
from agent.tools import knowledge_base_search_tool, KnowledgeBaseSearchInput

# Search for medical information
input_data = KnowledgeBaseSearchInput(query="menopause symptoms", limit=10)
results = await knowledge_base_search_tool(input_data)
```

### Retrieving User History
```python
from agent.tools import episodic_memory_search_tool, EpisodicSearchInput

# Search user's conversation history
input_data = EpisodicSearchInput(
    query="hot flashes discussed",
    user_id="user_123",
    session_id="session_456",
    limit=5
)
results = await episodic_memory_search_tool(input_data)
```

### Finding Entity Relationships
```python
from agent.tools import get_entity_relationships_tool, EntityRelationshipInput

# Explore medical entity connections
input_data = EntityRelationshipInput(entity_name="estrogen", depth=2)
relationships = await get_entity_relationships_tool(input_data)
```

## Environment Configuration

Required environment variables:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Database Provider
DB_PROVIDER=supabase  # or postgres
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key

# LLM Configuration
LLM_PROVIDER=gemini
LLM_API_KEY=your-api-key
LLM_CHOICE=gemini-2.5-flash

# Embedding Configuration
EMBEDDING_PROVIDER=gemini
EMBEDDING_API_KEY=your-api-key
EMBEDDING_MODEL=gemini-embedding-001
```

## Testing

Run the comprehensive test suite:

```bash
python test_dual_search.py
```

This tests:
1. Direct Neo4j knowledge base queries
2. Graphiti episodic memory storage and retrieval
3. Comprehensive search combining all methods
4. Routing logic between search strategies

## Performance Considerations

### Knowledge Base (Neo4j Direct)
- **Pros**: Fast graph traversal, complex relationship queries
- **Cons**: Requires indexing for text search
- **Optimization**: Create indexes on frequently queried properties

### User History (Graphiti)
- **Pros**: Temporal reasoning, fact extraction, LLM-enhanced
- **Cons**: Slower due to LLM processing
- **Optimization**: Use lighter models for ingestion

### Vector Search (pgvector)
- **Pros**: Semantic similarity, fast approximate search
- **Cons**: Requires embedding generation
- **Optimization**: Use IVFFlat index, batch embeddings

## Security and Privacy

### Data Isolation
- Knowledge base: Shared medical information (group_id="0")
- User history: Isolated per user (group_id=user_id)
- No cross-contamination between users

### Access Control
- Service role keys for database access
- User-specific group_ids for data partitioning
- RLS policies in PostgreSQL for additional security

## Future Enhancements

1. **Caching Layer**: Add Redis for frequently accessed entities
2. **Query Optimization**: Implement query result caching
3. **Batch Processing**: Optimize bulk episode ingestion
4. **Analytics**: Add usage tracking and performance metrics
5. **Federated Search**: Combine results with relevance scoring

## Troubleshooting

### Common Issues

1. **Empty Knowledge Base Results**
   - Check if data was ingested directly to Neo4j
   - Verify entity names match exactly
   - Try case-insensitive search

2. **Graphiti Connection Errors**
   - Ensure Neo4j indices are built
   - Check LLM API credentials
   - Verify embedding dimensions match

3. **Routing Issues**
   - Check group_id values
   - Verify search tool selection
   - Review routing logic in tools.py

## Conclusion

The dual search architecture provides optimal performance by using:
- Direct Neo4j for structured medical knowledge
- Graphiti for temporal user interactions
- pgvector for semantic document search

This separation ensures fast knowledge base queries while maintaining rich user context through Graphiti's LLM-enhanced processing.