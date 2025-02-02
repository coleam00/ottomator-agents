from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import sys
import os
import base64
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize FastAPI app and OpenAI client
app = FastAPI()
security = HTTPBearer()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

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
    files: Optional[List[Dict[str, Any]]] = None

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

def process_files_to_string(files: Optional[List[Dict[str, Any]]]) -> str:
    """Convert a list of files with base64 content into a formatted string."""
    if not files:
        return ""
        
    file_content = "File content to use as context:\n\n"
    for i, file in enumerate(files, 1):
        decoded_content = base64.b64decode(file['base64']).decode('utf-8')
        file_content += f"{i}. {file['name']}:\n\n{decoded_content}\n\n"
    return file_content        

@app.post("/api/file-agent", response_model=AgentResponse)
async def file_agent(
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
            msg_type = "user" if msg_data["type"] == "human" else "assistant" # Type conversion for OpenAI API
            msg_content = msg_data["content"]
            
            # Process files if they exist in the message data
            if msg_type == "user" and "data" in msg_data and "files" in msg_data["data"]:
                files_content = process_files_to_string(msg_data["data"]["files"])
                if files_content:
                    msg_content = f"{files_content}\n\n{msg_content}"
            
            messages.append({"role": msg_type, "content": msg_content})

        # Store user's query with files if present
        message_data = {"request_id": request.request_id}
        if request.files:
            message_data["files"] = request.files

        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query,
            data=message_data
        )

        # Prepare current message with files for OpenAI
        current_message = request.query
        if request.files:
            files_content = process_files_to_string(request.files)
            current_message = f"{files_content}\n\n{current_message}"

        # Prepare messages for OpenAI
        openai_messages = []
        openai_messages.extend(messages)  # Add conversation history
        openai_messages.append({"role": "user", "content": current_message})

        # Get response from OpenAI
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages
        )
        agent_response = completion.choices[0].message.content

        # Store agent's response
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=agent_response,
            data={"request_id": request.request_id}
        )

        return AgentResponse(success=True)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Store error message in conversation
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id}
        )
        return AgentResponse(success=False)

if __name__ == "__main__":
    import uvicorn
    # Feel free to change the port here if you need
    uvicorn.run(app, host="0.0.0.0", port=8001)
