# MCP Agent Army Tests

This directory contains tests for the MCP Agent Army project, including the dynamic model selection feature.

## Test Structure

- `test_model_selection.py`: Tests for the dynamic model selection feature in the CLI version
- `test_model_selection_api.py`: Tests for the API endpoints that support dynamic model selection

## Running Tests

### Option 1: Using run_tests.py

The simplest way to run tests is using the provided script:

```bash
python run_tests.py
```

### Option 2: Using pytest directly

You can also run pytest directly:

```bash
# Install dev dependencies first
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run specific test files
pytest tests/test_model_selection.py
pytest tests/test_model_selection_api.py

# Run with verbose output
pytest -v

# Run tests and show print output
pytest -v -s
```

## Writing New Tests

When adding new features, please add corresponding tests following these guidelines:

1. Create a new test file in this directory with the `test_` prefix
2. Use pytest fixtures for common setup
3. Mock external dependencies to avoid actual API calls
4. Test both success paths and error handling
5. Follow the existing test patterns for consistency

## Testing the Studio Integration Version

The tests for the Studio Integration version simulate API calls using FastAPI's TestClient. 
These tests do not require actually running the API server, as they use the TestClient
to make in-memory requests to the FastAPI application. 