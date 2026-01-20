"""
Ultimate Orchestrator - FastAPI Web Server
===========================================
Production-ready REST API for the Ultimate Orchestrator system.
Provides endpoints for agent orchestration, voice capabilities, and monitoring.
"""

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import os
from datetime import datetime
import uvicorn

from meta_orchestrator import MetaOrchestrator
from voice_interface import VoiceAgentFactory, VoiceConfig
from monitoring_dashboard import get_dashboard, Metric, MetricType
from unified_config import get_config
from livekit_voice import LiveKitIntegration, VoiceWebSocket

# Initialize FastAPI app
app = FastAPI(
    title="Ultimate Orchestrator API",
    description="Cloud-ready AI Agent Orchestration Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
config = get_config()
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator and monitoring
orchestrator = MetaOrchestrator()
dashboard = get_dashboard()

# Initialize LiveKit voice integration
livekit = LiveKitIntegration()
voice_ws = VoiceWebSocket()

# Mount static files for frontend
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    print(f"✅ Frontend mounted at /static from {frontend_path}")

# Request/Response Models
class AgentRequest(BaseModel):
    """Request to execute an agent task"""
    task: str = Field(..., description="Task description for the agent")
    agent_name: Optional[str] = Field(None, description="Specific agent to use (optional)")
    enable_voice: bool = Field(False, description="Enable voice capabilities")
    parallel: bool = Field(False, description="Enable parallel execution if multiple agents")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class AgentResponse(BaseModel):
    """Response from agent execution"""
    success: bool
    result: str
    agent_used: Optional[str] = None
    execution_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

class AgentDiscoveryRequest(BaseModel):
    """Request to discover suitable agents"""
    task_description: str
    required_capabilities: Optional[List[str]] = None

class VoiceRequest(BaseModel):
    """Request for voice conversation"""
    initial_message: Optional[str] = None
    agent_name: Optional[str] = None
    voice_config: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "orchestrator": "running",
            "monitoring": "running"
        }
    }

# Root endpoint - Serve frontend
@app.get("/")
async def root():
    """Serve the futuristic frontend interface"""
    frontend_index = Path(__file__).parent / "frontend" / "index.html"

    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    else:
        # Fallback to API info if frontend not available
        return {
            "name": "Ultimate Orchestrator API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "orchestrate": "/api/v1/orchestrate",
                "discover": "/api/v1/discover",
                "agents": "/api/v1/agents",
                "monitoring": "/api/v1/monitoring",
                "voice": "/api/v1/voice"
            }
        }

# Agent orchestration endpoint
@app.post("/api/v1/orchestrate", response_model=AgentResponse)
async def orchestrate_task(request: AgentRequest, background_tasks: BackgroundTasks):
    """
    Orchestrate an AI agent task.

    This endpoint:
    1. Discovers the best agent(s) for the task
    2. Executes the agent(s)
    3. Returns the result
    4. Records metrics
    """
    start_time = datetime.now()

    try:
        # Record execution start
        dashboard.record_metric(Metric(
            timestamp=start_time,
            agent_name=request.agent_name or "orchestrator",
            metric_type=MetricType.AGENT_EXECUTION,
            value=1
        ))

        # Execute the task
        result = await orchestrator.process_request(
            request.task,
            enable_voice=request.enable_voice
        )

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        # Record success
        dashboard.record_metric(Metric(
            timestamp=datetime.now(),
            agent_name=request.agent_name or "orchestrator",
            metric_type=MetricType.SUCCESS,
            value=1
        ))

        # Record latency
        dashboard.record_metric(Metric(
            timestamp=datetime.now(),
            agent_name=request.agent_name or "orchestrator",
            metric_type=MetricType.LATENCY,
            value=execution_time
        ))

        return AgentResponse(
            success=True,
            result=result,
            agent_used=request.agent_name or "meta-orchestrator",
            execution_time_ms=execution_time
        )

    except Exception as e:
        # Record error
        dashboard.record_metric(Metric(
            timestamp=datetime.now(),
            agent_name=request.agent_name or "orchestrator",
            metric_type=MetricType.ERROR,
            value=1,
            metadata={"error_message": str(e)}
        ))

        raise HTTPException(status_code=500, detail=str(e))

# Agent discovery endpoint
@app.post("/api/v1/discover")
async def discover_agents(request: AgentDiscoveryRequest):
    """
    Discover the best agents for a task.

    Returns a ranked list of agents that can handle the task.
    """
    try:
        candidates = orchestrator.registry.find_best_agent(
            request.task_description,
            request.required_capabilities
        )

        return {
            "task": request.task_description,
            "recommended_agents": [
                {
                    "name": agent.name,
                    "category": agent.category.value,
                    "description": agent.description,
                    "confidence": agent.confidence_score,
                    "capabilities": agent.capabilities,
                    "requires_voice": agent.requires_voice,
                    "requires_mcp": agent.requires_mcp
                }
                for agent in candidates
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# List all agents
@app.get("/api/v1/agents")
async def list_agents():
    """List all available agents in the registry"""
    try:
        agents_by_category = {}

        for agent_name, agent_metadata in orchestrator.registry.agents.items():
            category = agent_metadata.category.value
            if category not in agents_by_category:
                agents_by_category[category] = []

            agents_by_category[category].append({
                "name": agent_metadata.name,
                "description": agent_metadata.description,
                "capabilities": agent_metadata.capabilities,
                "requires_voice": agent_metadata.requires_voice,
                "requires_mcp": agent_metadata.requires_mcp
            })

        return {
            "total_agents": len(orchestrator.registry.agents),
            "agents_by_category": agents_by_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/api/v1/monitoring/summary")
async def get_monitoring_summary():
    """Get overall system monitoring summary"""
    try:
        return dashboard.get_system_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/agent/{agent_name}")
async def get_agent_monitoring(agent_name: str):
    """Get monitoring data for a specific agent"""
    try:
        summary = dashboard.get_agent_summary(agent_name)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/top")
async def get_top_agents(metric: str = "executions", limit: int = 10):
    """Get top agents by a specific metric"""
    try:
        return {
            "metric": metric,
            "limit": limit,
            "top_agents": dashboard.get_top_agents(metric, limit)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/alerts")
async def get_alerts(limit: int = 20):
    """Get recent alerts"""
    try:
        return {
            "alerts": dashboard.get_recent_alerts(limit)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/dashboard")
async def get_dashboard_view():
    """Get text-based dashboard view"""
    try:
        return {
            "dashboard": dashboard.render_dashboard()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoint
@app.get("/api/v1/config/summary")
async def get_config_summary():
    """Get configuration summary"""
    try:
        return {
            "summary": config.get_config_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== LIVEKIT VOICE ENDPOINTS =====

@app.post("/api/v1/voice/create-session")
async def create_voice_session(user_id: Optional[str] = None):
    """Create a new LiveKit voice session"""
    try:
        result = await livekit.create_voice_session(user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/end-session")
async def end_voice_session(room_name: str):
    """End a voice session"""
    try:
        result = await livekit.end_voice_session(room_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/voice/active-sessions")
async def get_active_voice_sessions():
    """Get all active voice sessions"""
    try:
        return {"sessions": livekit.get_active_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for voice streaming
@app.websocket("/ws/voice/{user_id}")
async def voice_websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for browser-based voice communication"""
    await websocket.accept()
    await voice_ws.handle_connection(websocket, user_id)

# WebSocket endpoint for real-time updates
@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates"""
    await websocket.accept()

    try:
        while True:
            # Send monitoring update every 5 seconds
            summary = dashboard.get_system_summary()
            await websocket.send_json(summary)
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("=" * 80)
    print("🚀 Ultimate Orchestrator API Starting...")
    print("=" * 80)
    print(f"Total Agents: {len(orchestrator.registry.agents)}")
    print(f"Environment: {config.config.environment}")
    print(f"Monitoring: {'Enabled' if config.config.monitoring.enabled else 'Disabled'}")
    print("=" * 80)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("=" * 80)
    print("🛑 Ultimate Orchestrator API Shutting Down...")
    print("=" * 80)

    # Export metrics before shutdown
    try:
        dashboard.export_metrics("final_metrics.json")
        print("✅ Metrics exported")
    except Exception as e:
        print(f"❌ Failed to export metrics: {e}")

# Main entry point
if __name__ == "__main__":
    # Get configuration
    # Render uses PORT, but also support SERVER_PORT for backwards compatibility
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8000")))
    workers = int(os.getenv("WORKERS", "1"))  # Use 1 worker for free tier
    reload = os.getenv("ENVIRONMENT", "production") == "development"

    print("=" * 80)
    print(f"🚀 Starting Ultimate Orchestrator on {host}:{port}")
    print(f"📊 Workers: {workers}")
    print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print("=" * 80)

    # Run server
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )
