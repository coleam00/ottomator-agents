from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import requests
import sys
import os
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
security = HTTPBearer()

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

def interact_with_voiceflow(action: dict, session_id: str) -> dict:
    """
    Send an interaction request to Voiceflow Dialog API.
    NOTE: Right now this only supports text responses, but in
    the future it will support buttons, carousels, etc.
    That is why the payload can also be a dict and the action_type is dynamic.
    
    Args:
        action (dict): Type of action and payload to send to the Voiceflow Dialog API
        session_id (str): Session ID for the user
        
    Returns:
        dict: JSON response from the API
        
    Raises:
        requests.exceptions.RequestException: If the request fails
        KeyError: If the API key environment variable is not set
    """
    
    # Get API key from environment variable
    api_key = os.getenv('VOICEFLOW_AGENT_API_KEY')
    if not api_key:
        raise KeyError("VOICEFLOW_AGENT_API_KEY environment variable not set")
    
    # Construct the URL with the session ID
    base_url = "https://general-runtime.voiceflow.com/state/user"
    url = f"{base_url}/{session_id}/interact"
    
    # Prepare the request headers
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
        'Authorization': api_key
    }
    
    # Prepare the request body
    body = {
        "action": action,
        "config": {
            "tts": False,
            "stripSSML": True,
            "stopAll": False,
            "excludeTypes": [
                "block",
                "debug",
                "flow"
            ]
        }
    }
    
    # Make the POST request
    response = requests.post(
        url=f"{url}?logs=off",
        headers=headers,
        json=body
    )
    
    # Raise an exception for bad status codes
    response.raise_for_status()
    
    # Return the JSON response
    return response.json()        

@app.post("/api/sample-voiceflow-agent", response_model=AgentResponse)
async def sample_voiceflow_agent(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        # Process the query - it could be either plain text or JSON
        # Plain text is for base responses to the agent
        # JSON is for when buttons are selected for the Voiceflow agent
        user_message = request.query
        action = {"type": "text", "payload": user_message}
        
        if user_message.startswith("```json"):
            try:
                # Remove the ```json prefix and parse
                json_str = user_message.replace("```json", "").strip()
                parsed_data = json.loads(json_str)
                
                # Extract the text for storing and the request data for action
                user_message = parsed_data.get("text", "")
                action = parsed_data.get("data", {}).get("request", {}) or parsed_data.get("data", {}).get("action", {})
            except json.JSONDecodeError as e:
                print(e)
                # If JSON parsing fails, treat it as regular text
                pass

        # Store user's query
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=user_message
        )

        # Call the Voicflow Dialog API with the appropriate action
        agent_response = interact_with_voiceflow(
            action=action,
            session_id=request.session_id
        )

        # Store agent's response
        # If you need to send a part of the message outside of the Voiceflow agent's response,
        # do that in the content field. Otherwise the entire response is in the data field.
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="",
            data=agent_response
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
