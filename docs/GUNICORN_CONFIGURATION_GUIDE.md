# Gunicorn Configuration Guide

## Critical Configuration Fix Applied

### Issue Resolved
- **Problem**: Invalid `worker_class_kwargs` parameter causing Gunicorn startup failures
- **Impact**: Deployment blocker preventing application from starting
- **Solution**: Removed invalid configuration and implemented correct Uvicorn optimizations via environment variables

### Before (BROKEN)
```python
# This configuration is INVALID and will cause startup failures
worker_class_kwargs = {
    "loop": "auto",
    "http": "auto", 
    "ws": "auto",
    "lifespan": "auto"
}
```

### After (FIXED)
```python
# Invalid parameter removed - optimizations now configured via environment variables
# See render.yaml and .env.example for proper implementation
```

## Valid Gunicorn Configuration Options

Our current `gunicorn.conf.py` uses only valid Gunicorn settings:

### Server Configuration
- `bind` - Server socket binding
- `workers` - Number of worker processes  
- `worker_class` - Worker class (uvicorn.workers.UvicornWorker)
- `worker_connections` - Connections per worker

### Performance Settings
- `max_requests` - Requests before recycling worker
- `max_requests_jitter` - Randomization for worker restart
- `timeout` - Worker timeout
- `keepalive` - Keep-alive timeout
- `graceful_timeout` - Graceful shutdown timeout

### Logging & Process Management
- `accesslog`, `errorlog`, `loglevel` - Logging configuration
- `proc_name` - Process name
- `preload_app` - Application preloading
- `worker_tmp_dir` - Temporary directory

## Uvicorn Optimization Implementation

### Correct Method: Environment Variables
Set these in your deployment environment:

```bash
UVICORN_LOOP=auto      # Optimal event loop selection
UVICORN_HTTP=auto      # Optimal HTTP implementation  
UVICORN_LIFESPAN=auto  # ASGI lifespan protocol handling
```

### Deployment Configuration
These are automatically set in:
- `render.yaml` - Production deployment
- `render.alternative.yaml` - Backup deployment
- `.env.example` - Local development template

### Performance Benefits
- **Uvloop Event Loop**: 2-4x performance improvement for I/O operations
- **HTTPTools Parser**: Faster HTTP request/response processing
- **Auto Selection**: Best implementation chosen automatically
- **Medical RAG Optimized**: Enhanced for vector search and streaming responses

## Common Configuration Mistakes to Avoid

### ❌ Invalid Approaches
```python
# These will cause startup failures:
worker_class_kwargs = {...}          # Not supported by Gunicorn
custom_worker_options = {...}        # Not a valid parameter
uvicorn_settings = {...}             # Not recognized
```

### ✅ Correct Approaches
```python
# Use standard Gunicorn parameters only
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"  
timeout = 180

# Configure Uvicorn via environment variables (not in config file)
```

## Configuration Validation

### Before Deployment
```bash
# Test configuration syntax
python -c "exec(open('gunicorn.conf.py').read())"

# Test FastAPI import
python -c "from agent.api import app; print('✅ Import successful')"
```

### During Development
- Use `.env.example` as template for local environment
- Test with both direct uvicorn and gunicorn+uvicorn
- Validate health endpoint responds correctly

## Troubleshooting

### If Deployment Still Fails
1. Check `render.alternative.yaml` which bypasses Gunicorn entirely
2. Use direct uvicorn execution: `python -m uvicorn agent.api:app --host 0.0.0.0 --port $PORT`
3. Verify all environment variables are set correctly
4. Check application logs for import errors

### Configuration Validation
- All parameters in `gunicorn.conf.py` must be valid Gunicorn settings
- Uvicorn optimizations must use environment variables only
- Module path must be correct: `agent.api:app`

## Medical RAG Specific Optimizations

Our configuration is optimized for:
- **Vector Search Operations**: Extended timeout (180s)
- **3072-Dimensional Embeddings**: Memory management settings
- **Streaming Responses**: Proper keepalive configuration  
- **Async Database Operations**: ASGI worker optimization
- **Knowledge Graph Queries**: Sufficient worker resources

## Team Guidelines

1. **Never add `worker_class_kwargs`** to gunicorn.conf.py
2. **Use environment variables** for Uvicorn optimizations
3. **Test configuration** before deployment
4. **Validate syntax** with Python exec()
5. **Monitor startup logs** during deployment

This guide ensures reliable deployment without configuration-related startup failures.