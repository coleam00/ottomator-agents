.PHONY: help install test type-check security clean run docker-build docker-up docker-down

# Default target
help:
	@echo "Medical RAG Agent - Simple Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run          Run the API server"
	@echo "  make ingest       Run document ingestion"
	@echo "  make cli          Run the CLI interface"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run tests"
	@echo "  make type-check   Run type checking"
	@echo "  make security     Run security checks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start services"
	@echo "  make docker-down  Stop services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Clean generated files"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Development
run:
	python -m agent.api

ingest:
	python -m ingestion.ingest --verbose

cli:
	python cli.py

# Testing
test:
	@if [ -d "tests" ]; then \
		pytest tests/ -v --tb=short || true; \
	else \
		echo "No tests directory found"; \
	fi

type-check:
	pip install mypy types-requests
	mypy agent/ ingestion/ --ignore-missing-imports --python-version 3.11 || true

security:
	pip install safety bandit
	safety check || true
	bandit -r agent/ ingestion/ -ll || true

# Docker
docker-build:
	docker build -t medical-rag-agent:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -f .coverage