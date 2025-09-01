"""
FastAPI endpoints for the agentic RAG system.
"""

import os
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from dotenv import load_dotenv

from .agent import rag_agent, AgentDependencies
from .unified_db_utils import (
    initialize_database,
    close_database,
    create_session,
    get_session,
    add_message,
    get_session_messages,
    test_connection,
    health_check,
    get_provider_info,
    validate_configuration
)
from .graph_utils import initialize_graph, close_graph, test_graph_connection
from .models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    StreamDelta,
    ErrorResponse,
    HealthStatus,
    ToolCall
)
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    list_documents_tool,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentListInput
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Application configuration
APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
# Import safe parsing function from models
from .models import _safe_parse_int
# Render uses PORT environment variable, fallback to APP_PORT, then default
APP_PORT = _safe_parse_int("PORT", _safe_parse_int("APP_PORT", 8000, min_value=1, max_value=65535), min_value=1, max_value=65535)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Background task management
background_tasks: set = set()  # Use set for easier management
EPISODIC_MEMORY_TIMEOUT = float(os.getenv("EPISODIC_MEMORY_TIMEOUT", "30.0"))  # seconds
EPISODIC_MEMORY_ASYNC = os.getenv("EPISODIC_MEMORY_ASYNC", "true").lower() == "true"  # Make it configurable

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set debug level for our module during development
if APP_ENV == "development":
    logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger.info("Starting up agentic RAG API...")
    
    try:
        # Initialize database connections
        await initialize_database()
        logger.info("Database initialized")
        
        # Initialize graph database
        await initialize_graph()
        logger.info("Graph database initialized")
        
        # Test connections
        db_ok = await test_connection()
        graph_ok = await test_graph_connection()
        
        if not db_ok:
            logger.error("Database connection failed")
        if not graph_ok:
            logger.error("Graph database connection failed")
        
        logger.info("Agentic RAG API startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down agentic RAG API...")
    
    try:
        # Cancel all background tasks
        for task in background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        if background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*background_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some background tasks did not complete in time")
        
        await close_database()
        await close_graph()
        logger.info("Connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI app
app = FastAPI(
    title="Agentic RAG with Knowledge Graph",
    description="AI agent combining vector search and knowledge graph for tech company analysis",
    version="0.1.0",
    lifespan=lifespan
)

# Add middleware with flexible CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Helper functions for agent execution
async def get_or_create_session(request: ChatRequest) -> str:
    """Get existing session or create new one.
    
    Handles both UUID format and legacy timestamp-based session IDs.
    """
    if request.session_id:
        # Check if it's a UUID or legacy format
        is_uuid = False
        try:
            from uuid import UUID
            UUID(request.session_id)  # This will raise if invalid
            is_uuid = True
        except (ValueError, TypeError):
            # Not a UUID - could be legacy format like "session-1756521395616"
            pass
        
        # For UUID format, check if session exists
        if is_uuid:
            session = await get_session(request.session_id)
            if session:
                return request.session_id
        else:
            # For legacy format, log but accept it temporarily
            # This allows existing clients to continue working
            logger.info(f"Accepting legacy session_id format: {request.session_id}")
            # Convert to a deterministic UUID based on the legacy ID
            import hashlib
            from uuid import UUID
            # Create a deterministic UUID from the legacy ID
            hash_digest = hashlib.md5(request.session_id.encode()).hexdigest()
            new_uuid = str(UUID(hash_digest))
            
            # Check if this converted session exists
            session = await get_session(new_uuid)
            if session:
                logger.debug(f"Found existing session for legacy ID {request.session_id} -> {new_uuid}")
                return new_uuid
            else:
                # Create new session with the converted UUID
                logger.info(f"Creating new session for legacy ID {request.session_id} -> {new_uuid}")
                await create_session(
                    session_id=new_uuid,  # Use the deterministic UUID
                    user_id=request.user_id,
                    metadata={**request.metadata, "legacy_id": request.session_id} if request.metadata else {"legacy_id": request.session_id}
                )
                return new_uuid
    
    # Create new session with proper UUID
    new_session_id = await create_session(
        user_id=request.user_id,
        metadata=request.metadata
    )
    
    # Log the new session ID for debugging
    logger.info(f"Created new session: {new_session_id}")
    
    return new_session_id


async def get_conversation_context(
    session_id: str,
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Get recent conversation context.
    
    Args:
        session_id: Session ID
        max_messages: Maximum number of messages to retrieve
    
    Returns:
        List of messages
    """
    messages = await get_session_messages(session_id, limit=max_messages)
    
    return [
        {
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages
    ]


def extract_tool_calls(result) -> List[ToolCall]:
    """
    Extract tool calls from Pydantic AI result.
    
    Args:
        result: Pydantic AI result object
    
    Returns:
        List of ToolCall objects
    """
    tools_used = []
    
    try:
        # Get all messages from the result
        messages = result.all_messages()
        
        for message in messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    # Check if this is a tool call part
                    if part.__class__.__name__ == 'ToolCallPart':
                        try:
                            # Debug logging to understand structure
                            logger.debug(f"ToolCallPart attributes: {dir(part)}")
                            logger.debug(f"ToolCallPart content: tool_name={getattr(part, 'tool_name', None)}")
                            
                            # Extract tool information safely
                            tool_name = str(part.tool_name) if hasattr(part, 'tool_name') else 'unknown'
                            
                            # Get args - the args field is a JSON string in Pydantic AI
                            tool_args = {}
                            if hasattr(part, 'args') and part.args is not None:
                                if isinstance(part.args, str):
                                    # Args is a JSON string, parse it
                                    try:
                                        import json
                                        tool_args = json.loads(part.args)
                                        logger.debug(f"Parsed args from JSON string: {tool_args}")
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"Failed to parse args JSON: {e}")
                                        tool_args = {}
                                elif isinstance(part.args, dict):
                                    tool_args = part.args
                                    logger.debug(f"Args already a dict: {tool_args}")
                            
                            # Alternative: use args_as_dict method if available
                            if hasattr(part, 'args_as_dict'):
                                try:
                                    tool_args = part.args_as_dict()
                                    logger.debug(f"Got args from args_as_dict(): {tool_args}")
                                except:
                                    pass
                            
                            # Get tool call ID
                            tool_call_id = None
                            if hasattr(part, 'tool_call_id'):
                                tool_call_id = str(part.tool_call_id) if part.tool_call_id else None
                            
                            # Create ToolCall with explicit field mapping
                            tool_call_data = {
                                "tool_name": tool_name,
                                "args": tool_args,
                                "tool_call_id": tool_call_id
                            }
                            logger.debug(f"Creating ToolCall with data: {tool_call_data}")
                            tools_used.append(ToolCall(**tool_call_data))
                        except Exception as e:
                            logger.debug(f"Failed to parse tool call part: {e}")
                            continue
    except Exception as e:
        logger.warning(f"Failed to extract tool calls: {e}")
    
    return tools_used


async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Optional[Dict[str, Any]] = None,
    tools_used: Optional[List[ToolCall]] = None
):
    """
    Save a conversation turn to the database and create episodic memory.
    
    Args:
        session_id: Session ID
        user_message: User's message
        assistant_message: Assistant's response
        metadata: Optional metadata
        tools_used: List of tools used in the conversation
    """
    # Save user message
    await add_message(
        session_id=session_id,
        role="user",
        content=user_message,
        metadata=metadata or {}
    )
    
    # Save assistant message
    await add_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message,
        metadata=metadata or {}
    )
    
    # Create episodic memory asynchronously with proper task management
    try:
        from .episodic_memory import episodic_memory_service
        
        # Convert tools_used to dict format
        tools_dict = None
        if tools_used:
            tools_dict = [{"tool_name": t.tool_name} for t in tools_used]
        
        # Create managed background task with timeout
        task = asyncio.create_task(
            _create_episodic_memory_with_timeout(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                tools_dict=tools_dict,
                metadata=metadata
            )
        )
        
        # Add to background tasks list for lifecycle management
        global background_tasks
        background_tasks.append(task)
        
        # Clean up completed tasks periodically
        background_tasks = [t for t in background_tasks if not t.done()]
        
        logger.debug(f"Initiated managed episodic memory creation for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to create episodic memory task: {e}")
        # Don't fail the conversation if episodic memory fails


async def _create_episodic_memory_with_timeout(
    session_id: str,
    user_message: str,
    assistant_message: str,
    tools_dict: Optional[List[Dict[str, Any]]],
    metadata: Optional[Dict[str, Any]]
):
    """Create episodic memory with timeout and error handling - can run async or sync."""
    try:
        from .episodic_memory import episodic_memory_service
        
        async def create_memory():
            """Inner function to create episodic memory."""
            try:
                await asyncio.wait_for(
                    episodic_memory_service.create_conversation_episode(
                        session_id=session_id,
                        user_message=user_message,
                        assistant_response=assistant_message,
                        tools_used=tools_dict,
                        metadata=metadata
                    ),
                    timeout=EPISODIC_MEMORY_TIMEOUT
                )
                logger.info(f"Successfully created episodic memory for session {session_id}")
            except asyncio.TimeoutError:
                logger.error(f"Episodic memory creation timed out after {EPISODIC_MEMORY_TIMEOUT}s for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to create episodic memory for session {session_id}: {e}")
        
        if EPISODIC_MEMORY_ASYNC:
            # Create background task - won't block response
            task = asyncio.create_task(create_memory())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
            logger.debug(f"Episodic memory creation scheduled in background for session {session_id}")
        else:
            # Run synchronously - will wait for completion
            await create_memory()
            
    except Exception as e:
        logger.error(f"Failed to create episodic memory task for session {session_id}: {e}")


async def get_episodic_context(
    session_id: str,
    user_id: Optional[str] = None,
    current_message: str = "",
    max_results: int = 5
) -> Optional[str]:
    """
    Retrieve relevant episodic memory context from Graphiti.
    
    This function searches for relevant historical context including:
    - Previous conversations in this session
    - User preferences and medical history
    - Related topics and entities
    
    Args:
        session_id: Current session ID
        user_id: User ID for personalized context
        current_message: Current user message to find relevant context
        max_results: Maximum number of memories to retrieve
    
    Returns:
        Formatted context string or None if no relevant context found
    """
    try:
        from .episodic_memory import episodic_memory_service
        
        # Skip if episodic memory is disabled
        if not episodic_memory_service._enabled:
            return None
        
        context_parts = []
        
        # 1. Search for session-specific memories
        session_memories = await episodic_memory_service.search_episodic_memories(
            query=current_message,
            session_id=session_id,
            user_id=user_id,
            limit=max_results
        )
        
        if session_memories:
            # Format session memories
            session_context = []
            for memory in session_memories[:3]:  # Limit to top 3 most relevant
                if 'fact' in memory:
                    session_context.append(f"- {memory['fact']}")
            
            if session_context:
                context_parts.append("Session Memory:\n" + "\n".join(session_context))
        
        # 2. If user_id provided, search for user-specific memories across sessions
        if user_id:
            user_memories = await episodic_memory_service.get_user_memories(
                user_id=user_id,
                limit=max_results
            )
            
            if user_memories:
                # Extract relevant facts about user preferences and history
                user_context = []
                for memory in user_memories[:3]:
                    if 'fact' in memory:
                        # Check if this memory is from a different session
                        memory_session = memory.get('source_node_uuid', '')
                        if session_id not in memory_session:
                            user_context.append(f"- {memory['fact']}")
                
                if user_context:
                    context_parts.append("User History:\n" + "\n".join(user_context))
        
        # 3. Search for medical entities and symptoms if mentioned
        if any(keyword in current_message.lower() for keyword in ['symptom', 'pain', 'condition', 'treatment', 'medication']):
            # Extract medical context
            medical_memories = await episodic_memory_service.search_episodic_memories(
                query=f"symptoms conditions treatments {current_message}",
                user_id=user_id,
                limit=max_results
            )
            
            if medical_memories:
                medical_context = []
                for memory in medical_memories[:2]:
                    if 'fact' in memory:
                        medical_context.append(f"- {memory['fact']}")
                
                if medical_context:
                    context_parts.append("Medical Context:\n" + "\n".join(medical_context))
        
        # Combine all context parts
        if context_parts:
            full_context = "\n\n".join(context_parts)
            logger.info(f"Retrieved episodic context for session {session_id}: {len(context_parts)} context sections")
            return full_context
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve episodic context: {e}")
        # Don't fail the request if episodic retrieval fails
        return None


# Simple in-memory cache for episodic context
_episodic_cache = {}
_cache_ttl = 300  # 5 minutes TTL

async def get_episodic_context_cached(
    session_id: str,
    user_id: Optional[str] = None,
    current_message: str = "",
    max_results: int = 5
) -> Optional[str]:
    """
    Cached version of get_episodic_context to avoid repeated queries.
    
    Args:
        session_id: Current session ID
        user_id: User ID for personalized context
        current_message: Current user message
        max_results: Maximum number of memories to retrieve
    
    Returns:
        Formatted context string or None
    """
    import time
    
    # Create cache key
    cache_key = f"{session_id}:{user_id}:{hash(current_message[:100])}"
    
    # Check cache
    if cache_key in _episodic_cache:
        cached_data, timestamp = _episodic_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            logger.debug(f"Using cached episodic context for key {cache_key}")
            return cached_data
    
    # Fetch fresh context
    context = await get_episodic_context(
        session_id=session_id,
        user_id=user_id,
        current_message=current_message,
        max_results=max_results
    )
    
    # Update cache
    _episodic_cache[cache_key] = (context, time.time())
    
    # Clean old cache entries periodically
    if len(_episodic_cache) > 100:
        current_time = time.time()
        _episodic_cache.clear()  # Simple cleanup - could be more sophisticated
    
    return context


async def execute_agent(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
    save_conversation: bool = True
) -> tuple[str, List[ToolCall]]:
    """
    Execute the agent with a message.
    
    Args:
        message: User message
        session_id: Session ID
        user_id: Optional user ID
        save_conversation: Whether to save the conversation
    
    Returns:
        Tuple of (agent response, tools used)
    """
    try:
        # Create dependencies
        deps = AgentDependencies(
            session_id=session_id,
            user_id=user_id
        )
        
        # Get conversation context from database (recent messages)
        db_context = await get_conversation_context(session_id)
        
        # Retrieve episodic memory context from Graphiti (with caching)
        episodic_context = await get_episodic_context_cached(
            session_id=session_id,
            user_id=user_id,
            current_message=message
        )
        
        # Build prompt with both contexts
        full_prompt = message
        context_parts = []
        
        # Add episodic memory context if available
        if episodic_context:
            context_parts.append(f"Historical Context:\n{episodic_context}")
        
        # Add recent conversation context
        if db_context:
            recent_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in db_context[-6:]  # Last 3 turns
            ])
            context_parts.append(f"Recent Conversation:\n{recent_context}")
        
        # Combine contexts with the current message
        if context_parts:
            full_prompt = "\n\n".join(context_parts) + f"\n\nCurrent question: {message}"
        
        # Run the agent
        result = await rag_agent.run(full_prompt, deps=deps)
        
        response = result.data
        tools_used = extract_tool_calls(result)
        
        # Save conversation if requested
        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=response,
                metadata={
                    "user_id": user_id,
                    "tool_calls": len(tools_used),
                    "had_episodic_context": bool(episodic_context)
                },
                tools_used=tools_used
            )
        
        return response, tools_used
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        error_response = f"I encountered an error while processing your request: {str(e)}"
        
        if save_conversation:
            await save_conversation_turn(
                session_id=session_id,
                user_message=message,
                assistant_message=error_response,
                metadata={"error": str(e)}
            )
        
        return error_response, []


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Agentic RAG with Knowledge Graph",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "search": "/search/*",
            "documents": "/documents",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.post("/api/users/register-neo4j")
async def register_neo4j_user(request: Request):
    """
    Register a user in Neo4j/Graphiti knowledge graph.
    This endpoint is kept for compatibility but registration happens automatically
    through group_id partitioning when the user creates their first episode.
    
    Expected payload:
    {
        "user_id": "uuid-string"
    }
    """
    try:
        # Parse request body
        data = await request.json()
        user_id = data.get("user_id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # With group_id partitioning, users are automatically isolated
        # No explicit registration needed - just return success
        logger.info(f"User {user_id} will be automatically isolated via group_id")
        return {
            "status": "success",
            "user_id": user_id,
            "message": "User isolation enabled via group_id partitioning"
        }
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Error in user registration endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthStatus)
async def health_check_endpoint():
    """Health check endpoint."""
    try:
        # Get comprehensive health check from unified utils
        health_data = await health_check()
        
        # Test graph connection
        graph_status = await test_graph_connection()
        
        # Determine overall status
        db_status = health_data.get("connection") == "ok"
        if db_status and graph_status:
            status = "healthy"
        elif db_status or graph_status:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthStatus(
            status=status,
            database=db_status,
            graph_database=graph_status,
            llm_connection=True,  # Assume OK if we can respond
            version="0.1.0",
            timestamp=datetime.now(),
            provider=health_data.get("provider", "unknown"),
            stats=health_data.get("stats", {})
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        # Get or create session
        session_id = await get_or_create_session(request)
        
        # Execute agent
        response, tools_used = await execute_agent(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id
        )
        
        return ChatResponse(
            message=response,
            session_id=session_id,
            tools_used=tools_used,
            metadata={"search_type": str(request.search_type)}
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    try:
        # Get or create session
        session_id = await get_or_create_session(request)
        
        async def generate_stream():
            """Generate streaming response using agent.iter() pattern."""
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                
                # Create dependencies
                deps = AgentDependencies(
                    session_id=session_id,
                    user_id=request.user_id
                )
                
                # Get conversation context from database
                db_context = await get_conversation_context(session_id)
                
                # Retrieve episodic memory context from Graphiti
                episodic_context = await get_episodic_context_cached(
                    session_id=session_id,
                    user_id=request.user_id,
                    current_message=request.message
                )
                
                # Build input with both contexts
                full_prompt = request.message
                context_parts = []
                
                # Add episodic memory context if available
                if episodic_context:
                    context_parts.append(f"Historical Context:\n{episodic_context}")
                
                # Add recent conversation context
                if db_context:
                    recent_context = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in db_context[-6:]
                    ])
                    context_parts.append(f"Recent Conversation:\n{recent_context}")
                
                # Combine contexts with the current message
                if context_parts:
                    full_prompt = "\n\n".join(context_parts) + f"\n\nCurrent question: {request.message}"
                
                # Save user message immediately
                await add_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata={"user_id": request.user_id}
                )
                
                full_response = ""
                
                # Stream using agent.iter() pattern
                async with rag_agent.iter(full_prompt, deps=deps) as run:
                    async for node in run:
                        if rag_agent.is_model_request_node(node):
                            # Stream tokens from the model
                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta
                                    
                                    if isinstance(event, PartStartEvent) and event.part.part_kind == 'text':
                                        delta_content = event.part.content
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                        full_response += delta_content
                                        
                                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                        delta_content = event.delta.content_delta
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta_content})}\n\n"
                                        full_response += delta_content
                
                # Extract tools used from the final result
                result = run.result
                tools_used = extract_tool_calls(result)
                
                # Send tools used information
                if tools_used:
                    tools_data = [
                        {
                            "tool_name": tool.tool_name,
                            "args": tool.args,
                            "tool_call_id": tool.tool_call_id
                        }
                        for tool in tools_used
                    ]
                    yield f"data: {json.dumps({'type': 'tools', 'tools': tools_data})}\n\n"
                
                # Save assistant response
                await add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    metadata={
                        "streamed": True,
                        "tool_calls": len(tools_used),
                        "had_episodic_context": bool(episodic_context)
                    }
                )
                
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                error_chunk = {
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/vector")
async def search_vector(request: SearchRequest):
    """Vector search endpoint."""
    try:
        input_data = VectorSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await vector_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="vector",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/graph")
async def search_graph(request: SearchRequest):
    """Knowledge graph search endpoint."""
    try:
        input_data = GraphSearchInput(
            query=request.query
        )
        
        start_time = datetime.now()
        results = await graph_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            graph_results=results,
            total_results=len(results),
            search_type="graph",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid")
async def search_hybrid(request: SearchRequest):
    """Hybrid search endpoint."""
    try:
        input_data = HybridSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await hybrid_search_tool(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="hybrid",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents_endpoint(
    limit: int = 20,
    offset: int = 0
):
    """List documents endpoint."""
    try:
        input_data = DocumentListInput(limit=limit, offset=offset)
        documents = await list_documents_tool(input_data)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information."""
    try:
        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return ErrorResponse(
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=str(uuid.uuid4())
    )


# Additional endpoints for database provider information
@app.get("/provider/info")
async def get_provider_info_endpoint():
    """Get information about the current database provider."""
    try:
        return await get_provider_info()
    except Exception as e:
        logger.error(f"Failed to get provider info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get provider info")


@app.get("/provider/validate")
async def validate_provider_config():
    """Validate the current database provider configuration."""
    try:
        return await validate_configuration()
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate configuration")


@app.get("/database/stats")
async def get_database_stats_endpoint():
    """Get database statistics."""
    try:
        from .unified_db_utils import get_database_stats
        return await get_database_stats()
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database stats")


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower()
    )