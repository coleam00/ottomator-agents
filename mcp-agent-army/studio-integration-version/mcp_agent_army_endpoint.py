from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import httpx
import sys
import os

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)

from mcp_agent_army import get_mcp_agent_army, select_model_for_task, get_model

# Load environment variables
load_dotenv()

primary_agent = None
mcp_stack = None

# Define a lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    global primary_agent
    global mcp_stack
    
    # Initialize the primary agent and get the stack that keeps MCP servers alive
    primary_agent, mcp_stack = await get_mcp_agent_army()
    
    yield
    
    # Cleanup: close the MCP servers when the application shuts down
    await mcp_stack.aclose()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str
    auto_model_selection: bool = False

class AgentResponse(BaseModel):
    success: bool
    model_used: Optional[str] = None
    rephrased_query: Optional[str] = None

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True    

async def fetch_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Convert to list and reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the Supabase messages table."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data

    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

@app.post("/api/mcp-agent-army", response_model=AgentResponse)
async def mcp_agent_army(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        # Fetch conversation history
        conversation_history = await fetch_conversation_history(request.session_id)
        
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            msg_data = msg["message"]
            msg_type = msg_data["type"]
            msg_content = msg_data["content"]
            msg = ModelRequest(parts=[UserPromptPart(content=msg_content)]) if msg_type == "human" else ModelResponse(parts=[TextPart(content=msg_content)])
            messages.append(msg)

        # Store user's query
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query
        )        

        # Variables to track model selection
        processed_query = request.query
        selected_model = None
        model_selection_metadata = {}

        # Apply dynamic model selection if enabled
        if request.auto_model_selection:
            try:
                print(f"Applying auto model selection for session {request.session_id}")
                selected_model, processed_query = await select_model_for_task(request.query)
                # Update the primary agent's model
                primary_agent.model = get_model(selected_model)
                
                model_selection_metadata = {
                    "selected_model": selected_model,
                    "original_query": request.query,
                    "rephrased_query": processed_query
                }
                print(f"Selected model: {selected_model}")
                print(f"Rephrased query: {processed_query}")
            except Exception as e:
                print(f"Error in model selection: {e}")
                # Continue with original query and default model

        # Run the agent with conversation history and processed query
        result = await primary_agent.run(
            processed_query,
            message_history=messages
        )

        # Store agent's response with model selection metadata if used
        response_data = {"request_id": request.request_id}
        if selected_model:
            response_data["model_selection"] = model_selection_metadata
            
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=result.data,
            data=response_data
        )

        # Reset model to default if it was changed
        if selected_model:
            primary_agent.model = get_model()

        # Return success response
        return AgentResponse(
            success=True,
            model_used=selected_model,
            rephrased_query=processed_query if selected_model else None
        )

    except Exception as e:
        print(f"Error processing agent request: {str(e)}")
        # Store error message in conversation
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id}
        )
        return AgentResponse(success=False)

@app.post("/api/toggle-model-selection")
async def toggle_model_selection(
    session_id: str,
    enable: bool,
    authenticated: bool = Depends(verify_token)
):
    """Toggle the auto model selection feature for a session."""
    try:
        # Store a system message about the toggle
        message = f"Auto model selection {'enabled' if enable else 'disabled'}"
        await store_message(
            session_id=session_id,
            message_type="system",
            content=message,
            data={"auto_model_selection": enable}
        )
        
        return {"success": True, "auto_model_selection": enable}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
