"""
Optimized FastAPI endpoints with compression, caching, and performance monitoring.
"""

import os
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
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
from .cache_manager import cache_manager
from .performance_optimizer import (
    initialize_performance_optimizations,
    cleanup_performance_optimizations,
    get_performance_report,
    metrics,
    response_compressor,
    track_performance
)
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
# Import optimized tools
from .tools_optimized import (
    vector_search_tool_optimized,
    graph_search_tool_optimized,
    hybrid_search_tool_optimized,
    list_documents_tool_optimized,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentListInput
)

load_dotenv()

logger = logging.getLogger(__name__)

# Application configuration
APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
from .models import _safe_parse_int
APP_PORT = _safe_parse_int("PORT", _safe_parse_int("APP_PORT", 8000, min_value=1, max_value=65535), min_value=1, max_value=65535)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Performance configuration
ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
COMPRESSION_MIN_SIZE = int(os.getenv("COMPRESSION_MIN_SIZE", "1000"))
ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan context manager with performance optimizations."""
    # Startup
    logger.info("Starting optimized RAG API...")
    
    try:
        # Initialize database connections
        await initialize_database()
        logger.info("Database initialized")
        
        # Initialize graph database
        await initialize_graph()
        logger.info("Graph database initialized")
        
        # Initialize cache manager
        await cache_manager.initialize()
        logger.info("Cache manager initialized")
        
        # Initialize performance optimizations
        await initialize_performance_optimizations()
        logger.info("Performance optimizations initialized")
        
        # Test connections
        db_ok = await test_connection()
        graph_ok = await test_graph_connection()
        
        if not db_ok:
            logger.error("Database connection failed")
        if not graph_ok:
            logger.error("Graph database connection failed")
        
        logger.info("Optimized RAG API startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down optimized RAG API...")
    
    try:
        # Cleanup performance optimizations
        await cleanup_performance_optimizations()
        
        # Close cache manager
        await cache_manager.close()
        
        # Close databases
        await close_database()
        await close_graph()
        
        logger.info("Optimized shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create optimized FastAPI app
app = FastAPI(
    title="Optimized Medical RAG with Knowledge Graph",
    description="High-performance AI agent with caching, compression, and monitoring",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware stack
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Compression middleware
if ENABLE_COMPRESSION:
    app.add_middleware(GZipMiddleware, minimum_size=COMPRESSION_MIN_SIZE)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Custom middleware for performance tracking
@app.middleware("http")
async def performance_tracking_middleware(request: Request, call_next):
    """Track request performance metrics."""
    start_time = time.time()
    
    # Track request
    metrics.request_count += 1
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    metrics.response_times.append(response_time)
    
    # Add performance headers
    response.headers["X-Response-Time"] = f"{response_time:.3f}"
    response.headers["X-Cache-Status"] = "HIT" if getattr(request.state, "cache_hit", False) else "MISS"
    
    # Log slow requests
    if response_time > 1.0:
        logger.warning(f"Slow request: {request.method} {request.url.path} took {response_time:.2f}s")
    
    return response


# Helper functions with caching
@track_performance("session_management")
async def get_or_create_session_optimized(request: ChatRequest) -> str:
    """Optimized session management with caching."""
    if request.session_id:
        # Check cache first
        cache_key = f"session:{request.session_id}"
        cached_session = await cache_manager.get(cache_key)
        
        if cached_session:
            return request.session_id
        
        # Check database
        session = await get_session(request.session_id)
        if session:
            # Cache the session
            await cache_manager.set(cache_key, session, ttl=3600)
            return request.session_id
    
    # Create new session
    new_session_id = await create_session(
        user_id=request.user_id,
        metadata=request.metadata
    )
    
    logger.info(f"Created new session: {new_session_id}")
    return new_session_id


async def get_conversation_context_cached(
    session_id: str,
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """Get conversation context with caching."""
    cache_key = f"context:{session_id}:{max_messages}"
    
    # Check cache
    cached_context = await cache_manager.get(cache_key)
    if cached_context:
        return cached_context
    
    # Get from database
    messages = await get_session_messages(session_id, limit=max_messages)
    
    context = [
        {
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages
    ]
    
    # Cache for short period (conversation context changes frequently)
    await cache_manager.set(cache_key, context, ttl=60)
    
    return context


# Optimized endpoints
@app.get("/")
async def root():
    """Root endpoint with version and status."""
    return {
        "name": "Optimized Medical RAG with Knowledge Graph",
        "version": "2.0.0",
        "status": "operational",
        "performance": {
            "caching_enabled": True,
            "compression_enabled": ENABLE_COMPRESSION,
            "monitoring_enabled": ENABLE_PERFORMANCE_MONITORING
        },
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "search": "/search/*",
            "documents": "/documents",
            "performance": "/performance",
            "cache_stats": "/cache/stats",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthStatus)
async def health_check_optimized():
    """Enhanced health check with performance metrics."""
    try:
        health_data = await health_check()
        graph_status = await test_graph_connection()
        
        # Get cache and performance stats
        cache_stats = cache_manager.get_stats()
        perf_summary = metrics.get_summary()
        
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
            llm_connection=True,
            version="2.0.0",
            timestamp=datetime.now(),
            provider=health_data.get("provider", "unknown"),
            stats={
                **health_data.get("stats", {}),
                "cache": cache_stats,
                "performance": perf_summary
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
@track_performance("chat_endpoint")
async def chat_optimized(request: ChatRequest, response: Response):
    """Optimized chat endpoint with caching and compression."""
    try:
        # Get or create session
        session_id = await get_or_create_session_optimized(request)
        
        # Check if we have a cached response for this exact query
        cache_key = cache_manager._generate_key("chat_response", {
            "session_id": session_id,
            "message": request.message,
            "search_type": str(request.search_type)
        })
        
        cached_response = await cache_manager.get(cache_key)
        if cached_response:
            request.state.cache_hit = True
            metrics.cache_hits += 1
            logger.info(f"Chat response cache hit for session {session_id}")
            return ChatResponse(**cached_response)
        
        # Execute agent with optimized tools
        from .agent import execute_agent
        agent_response, tools_used = await execute_agent(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id
        )
        
        # Create response
        chat_response = ChatResponse(
            message=agent_response,
            session_id=session_id,
            tools_used=tools_used,
            metadata={"search_type": str(request.search_type)}
        )
        
        # Cache the response
        await cache_manager.set(
            cache_key,
            chat_response.dict(),
            ttl=300  # 5 minutes cache
        )
        
        # Compress response if large
        response_data = chat_response.dict()
        if ENABLE_COMPRESSION and len(json.dumps(response_data)) > COMPRESSION_MIN_SIZE:
            response.headers["Content-Encoding"] = "gzip"
            compressed = response_compressor.compress_response(response_data)
            return Response(content=compressed, media_type="application/json")
        
        return chat_response
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream_optimized(request: ChatRequest):
    """Optimized streaming chat endpoint."""
    try:
        session_id = await get_or_create_session_optimized(request)
        
        async def generate_stream():
            """Generate optimized streaming response."""
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
                
                # Create dependencies
                deps = AgentDependencies(
                    session_id=session_id,
                    user_id=request.user_id
                )
                
                # Get cached context
                db_context = await get_conversation_context_cached(session_id)
                
                # Build input
                full_prompt = request.message
                if db_context:
                    recent_context = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in db_context[-6:]
                    ])
                    full_prompt = f"Recent Conversation:\n{recent_context}\n\nCurrent question: {request.message}"
                
                # Save user message
                await add_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata={"user_id": request.user_id}
                )
                
                full_response = ""
                
                # Stream using agent
                async with rag_agent.iter(full_prompt, deps=deps) as run:
                    async for node in run:
                        if rag_agent.is_model_request_node(node):
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
                
                # Extract and send tools used
                from .api import extract_tool_calls
                result = run.result
                tools_used = extract_tool_calls(result)
                
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
                        "tool_calls": len(tools_used)
                    }
                )
                
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/vector")
@track_performance("vector_search_endpoint")
async def search_vector_optimized(request: SearchRequest):
    """Optimized vector search endpoint."""
    try:
        input_data = VectorSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await vector_search_tool_optimized(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=[r.dict() for r in results],
            total_results=len(results),
            search_type="vector",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/graph")
@track_performance("graph_search_endpoint")
async def search_graph_optimized(request: SearchRequest):
    """Optimized graph search endpoint."""
    try:
        input_data = GraphSearchInput(query=request.query)
        
        start_time = datetime.now()
        results = await graph_search_tool_optimized(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            graph_results=[r.dict() for r in results],
            total_results=len(results),
            search_type="graph",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid")
@track_performance("hybrid_search_endpoint")
async def search_hybrid_optimized(request: SearchRequest):
    """Optimized hybrid search endpoint."""
    try:
        input_data = HybridSearchInput(
            query=request.query,
            limit=request.limit
        )
        
        start_time = datetime.now()
        results = await hybrid_search_tool_optimized(input_data)
        end_time = datetime.now()
        
        query_time = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            results=[r.dict() for r in results],
            total_results=len(results),
            search_type="hybrid",
            query_time_ms=query_time
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
@track_performance("list_documents_endpoint")
async def list_documents_optimized(limit: int = 20, offset: int = 0):
    """Optimized document listing."""
    try:
        input_data = DocumentListInput(limit=limit, offset=offset)
        documents = await list_documents_tool_optimized(input_data)
        
        return {
            "documents": [d.dict() for d in documents],
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoints
@app.get("/performance")
async def get_performance_metrics():
    """Get current performance metrics."""
    return get_performance_report()


@app.get("/cache/stats")
async def get_cache_statistics():
    """Get cache statistics."""
    return cache_manager.get_stats()


@app.post("/cache/clear")
async def clear_cache(prefix: Optional[str] = None):
    """Clear cache entries."""
    if prefix:
        await cache_manager.clear_prefix(prefix)
        return {"message": f"Cleared cache entries with prefix: {prefix}"}
    else:
        await cache_manager.memory_cache.clear()
        return {"message": "Cleared all cache entries"}


@app.get("/performance/slow-queries")
async def get_slow_queries():
    """Get list of slow queries."""
    return {
        "slow_queries": metrics.slow_queries[-50:],  # Last 50 slow queries
        "total_slow_queries": len(metrics.slow_queries)
    }


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "agent.api_optimized:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower(),
        # Performance settings
        workers=4 if APP_ENV == "production" else 1,
        loop="uvloop",  # Use uvloop for better performance
        access_log=APP_ENV == "development"
    )