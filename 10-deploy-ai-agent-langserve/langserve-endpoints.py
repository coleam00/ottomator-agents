from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import os

# Load .env file
load_dotenv()

from runnable import get_runnable

app = FastAPI(
    title="LangServe AI Agent",
    version="1.0",
    description="LangGraph backend for the AI Agents Masterclass series agent.",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def main():
    # Fetch the AI Agent LangGraph runnable which generates the workouts
    runnable = get_runnable()

    # Create the Fast API route to invoke the runnable
    add_routes(
        app,
        runnable
    )

    # Start the API
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()