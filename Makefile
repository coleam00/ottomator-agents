.PHONY: help install dev-install test lint format type-check security clean run docker-build docker-up docker-down

PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit

# Default target
help:
	@echo "Medical RAG Agent - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev-install    Install development dependencies"
	@echo "  make pre-commit     Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make run           Run the API server"
	@echo "  make ingest        Run document ingestion"
	@echo "  make cli           Run the CLI interface"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-int      Run integration tests only"
	@echo "  make coverage      Run tests with coverage report"
	@echo "  make lint          Run all linters"
	@echo "  make format        Format code with black and isort"
	@echo "  make type-check    Run type checking with mypy"
	@echo "  make security      Run security checks"
	@echo "  make quality       Run all quality checks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start all services with docker-compose"
	@echo "  make docker-down   Stop all services"
	@echo "  make docker-logs   Show container logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Clean up generated files"
	@echo "  make clean-all     Clean everything including .venv"

# Installation targets
install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

dev-install: install
	$(PIP) install -r requirements-dev.txt 2>/dev/null || true
	$(PIP) install pytest pytest-cov pytest-asyncio pytest-xdist pytest-timeout
	$(PIP) install black isort ruff mypy bandit safety
	$(PIP) install pre-commit

pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit run --all-files

# Development targets
run:
	$(PYTHON) -m agent.api

ingest:
	$(PYTHON) -m ingestion.ingest --verbose

ingest-clean:
	$(PYTHON) -m ingestion.ingest --clean --verbose

cli:
	$(PYTHON) cli.py

# Testing targets
test:
	$(PYTEST) tests/ -v --tb=short

test-unit:
	$(PYTEST) tests/ -m unit -v --tb=short

test-int:
	$(PYTEST) tests/ -m integration -v --tb=short

test-fast:
	$(PYTEST) tests/ -m "not slow" -v --tb=short --maxfail=1

coverage:
	$(PYTEST) tests/ \
		--cov=agent \
		--cov=ingestion \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml \
		--cov-fail-under=70

coverage-html: coverage
	@echo "Opening coverage report in browser..."
	@python -m webbrowser htmlcov/index.html

# Code quality targets
format:
	$(BLACK) agent/ ingestion/ tests/ cli.py
	$(ISORT) agent/ ingestion/ tests/ cli.py

lint:
	$(RUFF) check agent/ ingestion/ tests/

lint-fix:
	$(RUFF) check --fix agent/ ingestion/ tests/

type-check:
	$(MYPY) agent/ ingestion/ \
		--python-version 3.11 \
		--ignore-missing-imports \
		--no-strict-optional

security:
	$(BANDIT) -r agent/ ingestion/ -ll
	safety check --json || true
	pip-audit --desc --fix --dry-run || true

quality: lint type-check security test

# Docker targets
docker-build:
	docker build -t medical-rag-agent:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-ingest:
	docker-compose --profile ingestion up ingestion

docker-clean:
	docker-compose down -v
	docker rmi medical-rag-agent:latest || true

# Database targets
db-setup:
	@echo "Setting up PostgreSQL with pgvector..."
	psql -d "$$DATABASE_URL" -f sql/schema.sql

db-reset:
	@echo "Warning: This will delete all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		psql -d "$$DATABASE_URL" -f sql/schema.sql; \
	fi

# Cleanup targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

clean-all: clean
	rm -rf venv/
	rm -rf .venv/
	rm -rf env/

# Development workflow shortcuts
dev: format lint type-check test

quick: format lint-fix test-fast

ci: quality coverage

# Environment setup
env-setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please update it with your settings."; \
	else \
		echo ".env file already exists."; \
	fi

# Release targets
version:
	@echo "Current version: $$(git describe --tags --always)"

changelog:
	@git log --pretty=format:"- %s (%h)" $$(git describe --tags --abbrev=0 HEAD^)..HEAD

release-patch:
	@echo "Creating patch release..."
	@bumpversion patch

release-minor:
	@echo "Creating minor release..."
	@bumpversion minor

release-major:
	@echo "Creating major release..."
	@bumpversion major