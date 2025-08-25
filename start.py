#!/usr/bin/env python3
"""
Backup startup script for Render deployment.
This ensures the FastAPI application can be started even if there are module path issues.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the FastAPI app
try:
    from agent.api import app
    print("✅ Successfully imported FastAPI app from agent.api")
    
    if __name__ == "__main__":
        import uvicorn
        port = int(os.getenv("PORT", os.getenv("APP_PORT", 8000)))
        host = os.getenv("APP_HOST", "0.0.0.0")
        
        print(f"🚀 Starting FastAPI server on {host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
            access_log=True
        )
        
except ImportError as e:
    print(f"❌ Failed to import FastAPI app: {e}")
    print("📁 Current working directory:", os.getcwd())
    print("🐍 Python path:", sys.path)
    print("📂 Directory contents:", os.listdir("."))
    
    # Try to find the agent module
    if os.path.exists("agent"):
        print("✅ Found agent directory")
        print("📂 Agent directory contents:", os.listdir("agent"))
        
        if os.path.exists("agent/api.py"):
            print("✅ Found agent/api.py file")
        else:
            print("❌ Missing agent/api.py file")
    else:
        print("❌ Missing agent directory")
    
    sys.exit(1)