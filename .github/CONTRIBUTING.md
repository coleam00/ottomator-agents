# Contributing to Medical RAG Agent

Thank you for your interest in contributing to the Medical RAG Agent project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept responsibility for mistakes

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use issue templates when available
3. Provide clear descriptions and steps to reproduce
4. Include relevant logs and error messages

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit with conventional commits (see below)
6. Push to your fork
7. Open a Pull Request

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/marypause/marypause_ai.git
cd marypause_ai
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Coding Standards

#### Python Style Guide

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Use type hints for all functions
- Write docstrings in Google style

#### Code Quality Checks

Before submitting, ensure your code passes all checks:

```bash
# Format code
black agent/ ingestion/ tests/
isort agent/ ingestion/ tests/

# Run linting
ruff check agent/ ingestion/

# Type checking
mypy agent/ ingestion/

# Security scan
bandit -r agent/ ingestion/

# Run tests
pytest tests/ --cov=agent --cov=ingestion
```

### Commit Messages

We use Conventional Commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or fixes
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes

Example:
```
feat(agent): add support for hybrid search

Implemented hybrid search combining vector and text similarity
with configurable weights.

Closes #123
```

### Testing

#### Writing Tests

- Place tests in `tests/` directory
- Follow the existing test structure
- Use pytest fixtures for common setup
- Mock external dependencies
- Aim for >80% code coverage

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/agent/test_models.py

# Run with coverage
pytest --cov=agent --cov=ingestion --cov-report=html

# Run specific test markers
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

### Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for development workflow changes
- Add docstrings to all public functions
- Include type hints
- Update API documentation if endpoints change

### Pull Request Process

1. **Pre-submission Checklist:**
   - [ ] Tests pass locally
   - [ ] Code follows style guide
   - [ ] Documentation updated
   - [ ] Commit messages follow convention
   - [ ] Branch is up to date with main

2. **PR Description:**
   - Describe what changes you made
   - Explain why the changes are needed
   - Reference related issues
   - Include screenshots for UI changes

3. **Review Process:**
   - Address reviewer feedback
   - Keep PR focused and reasonably sized
   - Be patient and respectful

### Security

- Never commit secrets or API keys
- Use environment variables for configuration
- Report security vulnerabilities privately
- Follow security best practices

### Getting Help

- Check documentation first
- Search existing issues
- Ask in discussions
- Join our community chat

## Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Project documentation

Thank you for contributing to Medical RAG Agent! 🎉