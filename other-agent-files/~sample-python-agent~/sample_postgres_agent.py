from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncpg
import json
import sys
import os

# Load environment variables
load_dotenv()

# Database connection pool
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_pool
    db_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
    yield
    # Shutdown
    if db_pool:
        await db_pool.close()

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

# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str

class AgentResponse(BaseModel):
    success: bool

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
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, created_at, session_id, message
                FROM messages 
                WHERE session_id = $1 
                ORDER BY created_at DESC 
                LIMIT $2
            """, session_id, limit)
            
            # Convert to list and reverse to get chronological order
            messages = [
                {
                    "id": str(row["id"]),
                    "created_at": row["created_at"].isoformat(),
                    "session_id": row["session_id"],
                    "message": row["message"]
                }
                for row in rows
            ]
            return messages[::-1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the messages table."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data

    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages (session_id, message)
                VALUES ($1, $2)
            """, session_id, json.dumps(message_obj))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

@app.post("/api/sample-postgres-agent", response_model=AgentResponse)
async def sample_postgres_agent(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        # Fetch conversation history from the DB
        conversation_history = await fetch_conversation_history(request.session_id)
        
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            msg_data = msg["message"]
            msg_type = msg_data["type"]
            msg_content = msg_data["content"]
            msg = {"role": msg_type, "content": msg_content}
            messages.append(msg)

        # Store user's query
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query
        )            

        """
        TODO:
        This is where you insert the custom logic to get the response from your agent.
        Your agent can also insert more records into the database to communicate
        actions/status as it is handling the user's question/request.
        Additionally:
            - Use the 'messages' array defined about for the chat history. This won't include the latest message from the user.
            - Use request.query for the user's prompt.
            - Use request.session_id if you need to insert more messages into the DB in the agent logic.
        """
        agent_response = "This is a sample agent response..."

        # Store agent's response
        await store_message(
            session_id=request.session_id,
            message_type="assistant",
            content=agent_response
        )

        return AgentResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
