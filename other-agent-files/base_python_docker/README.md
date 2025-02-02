# Base Python Docker Image for Ottomator Agents

This is the base Docker image used by Python-based Live Agent Studio agents. It provides a common foundation with all the necessary Python packages pre-installed.

## Features

- Python 3.11 with slim base image
- Common dependencies pre-installed (FastAPI, Uvicorn, Pydantic, etc.)
- Non-root user setup for security
- Port 8001 exposed by default

## Building the Image

```bash
docker build -t ottomator/base-python:latest .
```

## Using the Base Image

In your agent's Dockerfile, use this as your base image:

```dockerfile
FROM ottomator/base-python:latest

WORKDIR /app

# Copy your application code
COPY . .

# Add your custom commands here
CMD ["uvicorn", "your_app:app", "--host", "0.0.0.0", "--port", "8001"]
```

## Included Packages

See `requirements.txt` for the full list of pre-installed packages.
