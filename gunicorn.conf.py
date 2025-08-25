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
max_requests = 1200  # Slightly increased for better throughput
max_requests_jitter = 150  # Proportional jitter

# Thread workers for I/O bound tasks
threads = 2

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
timeout = 180  # Increased for embedding generation and vector search
keepalive = 5  # Improved for streaming responses
graceful_timeout = 30

# Memory management - prevents memory leaks
# max_requests and max_requests_jitter already defined above

# Worker processes lifecycle - use shared memory for better performance
worker_tmp_dir = "/dev/shm"

# Performance tuning
max_worker_connections = 1000  # Total connections across all workers
worker_recycle_limit = None

# FastAPI/ASGI specific optimizations
loop = "auto"  # Let uvicorn choose the best event loop
http = "auto"  # Let uvicorn choose the best HTTP implementation

# Enable access logging in production for monitoring
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'