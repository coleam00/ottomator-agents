# Episodic Memory Context Retrieval

## Overview

The episodic memory context retrieval system enables the AI agent to "remember" previous interactions and use that historical context to provide more personalized and contextually aware responses. This system retrieves relevant information from Graphiti (the knowledge graph) before processing each user query.

## Key Features

### 1. **Pre-Query Context Retrieval**
Before processing any user message, the system automatically:
- Searches for previous conversations in the current session
- Retrieves user preferences and medical history
- Identifies relevant topics and entities from past interactions
- Integrates this context into the agent's prompt

### 2. **Multi-Level Context**
The system retrieves context at three levels:
- **Session Memory**: Recent interactions within the current session
- **User History**: Cross-session memories for personalized responses
- **Medical Context**: Specific medical entities, symptoms, and treatments mentioned

### 3. **Efficient Caching**
- In-memory cache with 5-minute TTL to avoid repeated queries
- Cache key based on session ID, user ID, and message hash
- Automatic cache cleanup when size exceeds threshold

## Architecture

### Components

1. **`get_episodic_context()`** (agent/api.py)
   - Main function for retrieving episodic context
   - Searches Graphiti for relevant memories
   - Formats context into structured sections

2. **`get_episodic_context_cached()`** (agent/api.py)
   - Cached wrapper around `get_episodic_context()`
   - Reduces database queries for repeated requests
   - Manages cache lifecycle

3. **`EpisodicMemoryService`** (agent/episodic_memory.py)
   - Core service for managing episodic memories
   - Enhanced search with relevance scoring
   - Session and user-specific retrieval methods

### Data Flow

```
User Query → API Endpoint → get_episodic_context() → Graphiti Search
                ↓                     ↓
         Execute Agent ← Formatted Context ← Retrieved Memories
                ↓
         Agent Response (with historical context)
```

## Usage

### Basic Implementation

The system automatically retrieves context for every chat request:

```python
# In execute_agent() function
episodic_context = await get_episodic_context(
    session_id=session_id,
    user_id=user_id,
    current_message=message
)

# Context is integrated into the prompt
if episodic_context:
    full_prompt = f"Historical Context:\n{episodic_context}\n\nCurrent question: {message}"
```

### Testing Context Retrieval

Run the test script to verify the system is working:

```bash
python test_episodic_context.py
```

### Demo Application

See the system in action with the demo script:

```bash
# Start the API server
python -m agent.api

# In another terminal, run the demo
python examples/episodic_memory_demo.py
```

## Configuration

### Environment Variables

- `ENABLE_EPISODIC_MEMORY`: Enable/disable episodic memory (default: "true")
- `EPISODIC_BATCH_SIZE`: Number of memories to batch process (default: 5)
- `EPISODIC_MEMORY_TIMEOUT`: Timeout for memory creation (default: 30 seconds)
- `MEDICAL_ENTITY_EXTRACTION`: Enable medical entity extraction (default: "true")

### Cache Settings

```python
_cache_ttl = 300  # 5 minutes TTL
max_cache_size = 100  # Maximum cache entries
```

## Features in Detail

### 1. Session Memory Retrieval

The system searches for memories specific to the current session:

```python
session_memories = await episodic_memory_service.search_episodic_memories(
    query=current_message,
    session_id=session_id,
    user_id=user_id,
    limit=max_results
)
```

### 2. User History Retrieval

For returning users, the system retrieves cross-session memories:

```python
user_memories = await episodic_memory_service.get_user_memories(
    user_id=user_id,
    limit=max_results
)
```

### 3. Medical Context Extraction

When medical terms are detected, specialized context is retrieved:

```python
if any(keyword in message.lower() for keyword in ['symptom', 'pain', 'condition']):
    medical_memories = await episodic_memory_service.search_episodic_memories(
        query=f"symptoms conditions treatments {current_message}",
        user_id=user_id
    )
```

### 4. Relevance Scoring

Memories are scored and ranked by relevance:

- +20 points for session ID match
- +10 points for conversation episodes
- +5 points for recent memories
- +2 points per query term match

## Example Interactions

### Session Continuity

```
User: "I have a severe headache on the right side"
Assistant: "I understand you're experiencing a severe headache..."

[Memory stored: Patient has right-sided headache]

User: "What triggers should I avoid?"
Assistant: "Based on your right-sided headache symptoms, common migraine triggers to avoid include..."
```

### Cross-Session Memory

```
Session 1:
User: "I'm allergic to penicillin"
Assistant: "I've noted your penicillin allergy..."

Session 2 (same user):
User: "I need antibiotics for an infection"
Assistant: "Given your penicillin allergy from our previous conversation, I'd suggest alternatives like..."
```

## Benefits

1. **Improved Continuity**: Conversations feel more natural and connected
2. **Personalization**: Responses are tailored to individual user history
3. **Medical Safety**: Important medical information (allergies, conditions) is retained
4. **Reduced Repetition**: Users don't need to repeat information
5. **Context Awareness**: The agent understands the full conversation context

## Performance Considerations

- **Async Operations**: All retrieval is asynchronous to avoid blocking
- **Timeout Protection**: 30-second timeout on episodic memory operations
- **Graceful Degradation**: If retrieval fails, the system continues without context
- **Selective Retrieval**: Only relevant memories are fetched based on query content

## Monitoring and Debugging

### Logging

The system provides detailed logging:

```python
logger.info(f"Retrieved episodic context for session {session_id}: {len(context_parts)} context sections")
```

### Metadata Tracking

Each response includes metadata about context usage:

```json
{
  "metadata": {
    "had_episodic_context": true,
    "tool_calls": 2
  }
}
```

## Future Enhancements

1. **Semantic Clustering**: Group related memories for better context
2. **Importance Decay**: Weight recent memories more heavily
3. **Context Summarization**: Compress long histories into summaries
4. **User Preferences Learning**: Automatically identify and store preferences
5. **Multi-Modal Context**: Include image and document context

## Troubleshooting

### No Context Retrieved

1. Check if episodic memory is enabled: `ENABLE_EPISODIC_MEMORY=true`
2. Verify Graphiti connection is working
3. Ensure memories have been created for the session
4. Check logs for retrieval errors

### Slow Context Retrieval

1. Use the cached version: `get_episodic_context_cached()`
2. Reduce `max_results` parameter
3. Ensure Neo4j indices are properly configured
4. Monitor Graphiti query performance

### Incorrect Context

1. Verify user_id is consistent across requests
2. Check session_id format and consistency
3. Review relevance scoring in `search_episodic_memories()`
4. Ensure proper group_id isolation in Graphiti