# Gemini Code Assistant Context

## Project Overview

This project is a sophisticated medical Retrieval-Augmented Generation (RAG) agent. It leverages a hybrid approach combining semantic vector search with a temporal knowledge graph to provide accurate and context-aware answers to medical queries.

The system is built in Python and consists of three main components:

1.  **Document Ingestion Pipeline:** A robust pipeline that processes medical documents (in Markdown format), performs semantic chunking, generates vector embeddings, and builds a knowledge graph of entities and their relationships.
2.  **AI Agent:** A conversational agent built with Pydantic AI that can intelligently query the vector database and the knowledge graph to synthesize comprehensive answers.
3.  **FastAPI Server:** A high-performance API server that exposes the agent's capabilities through a RESTful interface, including support for streaming responses.

### Key Technologies

*   **Backend:** Python, FastAPI
*   **AI/ML:** Pydantic AI, Pydantic
*   **Databases:**
    *   **Vector Search:** PostgreSQL with `pgvector`
    *   **Knowledge Graph:** Neo4j with `Graphiti`
*   **Tooling:** Docker, Makefile, pytest, mypy, flake8, bandit

## Building and Running

The project uses a `Makefile` to streamline common development tasks.

### Installation

To install the necessary dependencies, run:

```bash
make install
```

This will install all packages listed in `requirements.txt`.

### Data Ingestion

Before running the agent, you must ingest the medical documents located in the `medical_docs/` directory.

```bash
make ingest
```

This command runs the ingestion pipeline (`ingestion/ingest.py`), which will process the documents, create embeddings, and populate both the PostgreSQL and Neo4j databases.

### Running the Application

To start the FastAPI server, run:

```bash
make run
```

The API will be available at `http://localhost:8058`.

### Command-Line Interface (CLI)

The project includes an interactive CLI for chatting with the agent. To use it, run the following command in a separate terminal:

```bash
make cli
```

### Running Tests

The project has a comprehensive test suite using `pytest`. To run the tests, use:

```bash
make test
```

## Development Conventions

*   **Dependency Management:** The project uses a `venv` for virtual environment management and `pip` with a `requirements.txt` file for dependencies.
*   **Code Style & Quality:**
    *   **Linting:** `flake8` and `pylint` are used for linting. Run `make lint`.
    *   **Type Checking:** `mypy` is used for static type checking. Run `make type-check`.
    *   **Security:** `safety` and `bandit` are used for security analysis. Run `make security`.
*   **Configuration:** The application is configured using a `.env` file in the project root. An example is provided in `.env.example`.
*   **Directory Structure:**
    *   `agent/`: Contains the core AI agent, API, and related modules.
    *   `ingestion/`: Contains the document ingestion pipeline.
    *   `sql/`: Contains SQL schema and migration files.
    *   `tests/`: Contains all the tests.
    *   `medical_docs/`: Contains the medical documents to be ingested.
*   **Docker:** The project includes a `Dockerfile` and `docker-compose.yml` for containerization.
    *   `make docker-build`: Builds the Docker image.
    *   `make docker-up`: Starts the application and services using Docker Compose.
    *   `make docker-down`: Stops the Docker Compose services.
