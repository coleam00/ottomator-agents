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

def check_deployment_configs():
    """Check deployment configuration files."""
    print("🔍 Checking deployment configuration files...")
    
    configs = [
        ('Procfile', 'Backup start command for Render'),
        ('render.yaml', 'Primary Render configuration'),
        ('runtime.txt', 'Python version specification'),
        ('gunicorn.conf.py', 'Gunicorn server configuration')
    ]
    
    all_good = True
    for filename, description in configs:
        if Path(filename).exists():
            print(f"✅ Found {filename} - {description}")
            
            # Special check for Procfile content
            if filename == "Procfile":
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    expected = "web: gunicorn --config gunicorn.conf.py agent.api:app"
                    if content == expected:
                        print(f"   ✅ Correct start command: {content}")
                    else:
                        print(f"   ⚠️  Start command: {content}")
                        print(f"   Expected: {expected}")
                        
        else:
            print(f"❌ Missing {filename} - {description}")
            if filename == "Procfile":
                print("   This is CRITICAL - Render may use wrong start command!")
                all_good = False
    
    return all_good

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
    
    # Check deployment configurations
    config_ok = check_deployment_configs()
    if not config_ok:
        print("❌ Deployment configuration issues detected")
        all_good = False
    print()
    
    # Final verdict
    if all_good:
        print("🚀 All deployment checks passed! Ready for production.")
        print("\n📋 Deployment Summary:")
        print("   - Procfile ensures correct start command")
        print("   - render.yaml provides full configuration")
        print("   - FastAPI app is importable")
        print("   - All critical modules available")
        return 0
    else:
        print("💥 Deployment checks failed. Please fix the issues above.")
        print("\n🔧 If Render still uses wrong start command:")
        print("   1. Check Procfile exists and has correct content")
        print("   2. Verify render.yaml is in repository root")
        print("   3. Configure start command manually in Render dashboard")
        return 1

if __name__ == "__main__":
    sys.exit(main())