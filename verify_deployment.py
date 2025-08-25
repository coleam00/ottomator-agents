#!/usr/bin/env python3
"""
Deployment verification script for the Medical RAG Agent system.
Checks all critical components before deployment.
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major != 3 or version.minor < 11:
        print(f"❌ Python version {version.major}.{version.minor} not supported. Requires Python 3.11+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_critical_modules():
    """Check if critical modules can be imported."""
    critical_modules = [
        'fastapi',
        'uvicorn',
        'gunicorn',
        'pydantic',
        'asyncpg',
        'neo4j',
        'openai',
        'anthropic'
    ]
    
    failed = []
    for module in critical_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            failed.append(module)
    
    return len(failed) == 0, failed

def check_project_structure():
    """Check if required project files exist."""
    required_files = [
        'agent/api.py',
        'agent/agent.py',
        'agent/models.py',
        'agent/tools.py',
        'requirements.txt',
        'gunicorn.conf.py'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found {file_path}")
        else:
            print(f"❌ Missing {file_path}")
            missing.append(file_path)
    
    return len(missing) == 0, missing

def check_fastapi_app():
    """Check if FastAPI app can be imported."""
    try:
        from agent.api import app
        print("✅ FastAPI app imported successfully")
        
        # Check if app is properly configured
        if hasattr(app, 'title') and app.title:
            print(f"✅ FastAPI app configured with title: {app.title}")
        else:
            print("⚠️ FastAPI app missing title configuration")
            
        return True, None
    except Exception as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        return False, str(e)

def check_environment_variables():
    """Check critical environment variables."""
    env_vars = {
        'PORT': 'Render port configuration',
        'APP_ENV': 'Application environment',
        'LOG_LEVEL': 'Logging configuration'
    }
    
    missing = []
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}={value} ({description})")
        else:
            print(f"⚠️ Missing {var} - {description}")
            missing.append(var)
    
    return missing

def main():
    """Run all deployment checks."""
    print("🔍 Verifying deployment readiness...\n")
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    print()
    
    # Check critical modules
    modules_ok, failed_modules = check_critical_modules()
    if not modules_ok:
        print(f"❌ Missing critical modules: {failed_modules}")
        all_good = False
    print()
    
    # Check project structure
    structure_ok, missing_files = check_project_structure()
    if not structure_ok:
        print(f"❌ Missing required files: {missing_files}")
        all_good = False
    print()
    
    # Check FastAPI app
    app_ok, app_error = check_fastapi_app()
    if not app_ok:
        print(f"❌ FastAPI app check failed: {app_error}")
        all_good = False
    print()
    
    # Check environment variables
    missing_env = check_environment_variables()
    if missing_env:
        print(f"⚠️ Missing environment variables: {missing_env}")
    print()
    
    # Final verdict
    if all_good:
        print("🚀 All deployment checks passed! Ready for production.")
        return 0
    else:
        print("💥 Deployment checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())