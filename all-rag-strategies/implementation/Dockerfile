# Dockerfile specifically for RAG Voice Agent
FROM python:3.11-slim

# Install system dependencies including PostgreSQL client
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install UV package manager
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies
RUN uv sync --frozen || uv sync

# Pre-download models for faster startup
RUN python -c "from livekit.plugins import silero; silero.VAD.load()" || true

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncpg; import sys; sys.exit(0)"

# Run the RAG agent
CMD ["uv", "run", "python", "rag_agent.py"]