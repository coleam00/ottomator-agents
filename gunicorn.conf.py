"""
Gunicorn configuration for FastAPI application with ASGI support
"""
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', os.getenv('APP_PORT', '10000'))}"
backlog = 2048

# Worker processes - optimized for Render's free tier and medical RAG workload
workers = int(os.getenv('WEB_CONCURRENCY', '2'))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000  # Connections per worker
max_requests = 1200  # Requests before recycling worker (prevents memory leaks)
max_requests_jitter = 150  # Randomization to prevent all workers restarting at once

# Thread workers for I/O bound tasks (only for sync workers, N/A for UvicornWorker)
# threads = 2  # Commented out - not applicable to ASGI workers

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv('LOG_LEVEL', 'info').lower()

# Process naming
proc_name = 'marypause-ai'

# Server mechanics
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None

# Application
module = "agent.api:app"

# Preload application
preload_app = True

# Timeout settings - optimized for medical RAG operations
timeout = 180  # Worker timeout for long-running operations (embedding generation, vector search)
keepalive = 5  # Keep-alive timeout for persistent connections (good for streaming)
graceful_timeout = 30  # Time to wait for workers to finish handling requests during shutdown

# Memory management - prevents memory leaks
# max_requests and max_requests_jitter already defined above

# Worker processes lifecycle - use shared memory for better performance
worker_tmp_dir = "/dev/shm"

# Performance tuning
max_worker_connections = 1000  # Total connections across all workers

# ASGI/Uvicorn Performance Optimizations
# NOTE: worker_class_kwargs is NOT supported by Gunicorn - removed to prevent startup failures
# 
# Uvicorn optimizations are configured via environment variables:
# These should be set in your deployment environment (render.yaml, .env, etc.)
#
# UVICORN_LOOP=auto     # Let uvicorn choose the best event loop (uvloop if available)
# UVICORN_HTTP=auto     # Let uvicorn choose the best HTTP implementation (httptools if available) 
# UVICORN_WS=auto       # WebSocket implementation (though not used in this API)
# UVICORN_LIFESPAN=auto # ASGI lifespan protocol handling
#
# Alternative command line method:
# gunicorn --worker-class uvicorn.workers.UvicornWorker \
#          --config gunicorn.conf.py \
#          --worker-class-kwargs '{"loop": "auto", "http": "auto"}' \
#          agent.api:app
#
# However, environment variables are the recommended approach for deployment compatibility.

# Enable access logging in production for monitoring
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'