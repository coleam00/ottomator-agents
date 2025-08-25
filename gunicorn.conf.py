"""
Gunicorn configuration for FastAPI application with ASGI support
"""
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', os.getenv('APP_PORT', '10000'))}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WEB_CONCURRENCY', '1'))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100

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

# Timeout
timeout = 120
keepalive = 2

# Memory management
max_requests = 1000
max_requests_jitter = 100

# Worker processes lifecycle
worker_tmp_dir = "/dev/shm"