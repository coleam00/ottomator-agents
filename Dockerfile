# Multi-stage build for Medical RAG Agent
FROM python:3.11-slim as builder

# Set build arguments
ARG VERSION=dev
ARG BUILD_DATE
ARG VCS_REF

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Labels
LABEL org.opencontainers.image.title="Medical RAG Agent" \
      org.opencontainers.image.description="Medical RAG Agent with Knowledge Graph capabilities" \
      org.opencontainers.image.url="https://github.com/marypause/marypause_ai" \
      org.opencontainers.image.source="https://github.com/marypause/marypause_ai" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT" \
      maintainer="MaryPause AI <contact@marypause.ai>"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Set Python path
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/medical_docs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT:-8058}/health || exit 1

# Expose port
EXPOSE ${APP_PORT:-8058}

# Default command
CMD ["python", "-m", "agent.api"]