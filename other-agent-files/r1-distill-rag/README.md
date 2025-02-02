# R1 Distill RAG System

This project showcases the power of DeepSeek's R1 model in an agentic RAG (Retrieval-Augmented Generation) system - built using Smolagents from HuggingFace. R1, known for its exceptional reasoning capabilities and instruction-following abilities, serves as the core reasoning engine. The system combines R1's strengths with efficient document retrieval and a separate conversation model to create a powerful, context-aware question-answering system.

## Setup

1. Clone the repository

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by copying `.env.example` to `.env`:
```bash
cp .env.example .env
```

5. Configure your `.env` file:

### Using HuggingFace (Cloud API)
```env
USE_HUGGINGFACE=yes
HUGGINGFACE_API_TOKEN=your_token_here
REASONING_MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
TOOL_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
```

### Using Ollama (Local Inference)
```env
USE_HUGGINGFACE=no
HUGGINGFACE_API_TOKEN=
REASONING_MODEL_ID=deepseek-r1:7b-8k
TOOL_MODEL_ID=qwen2.5:14b-instruct-8k
```

## Setting Up Ollama Models

The following models are recommended examples that provide a good balance of performance and resource usage, but you can use any compatible Ollama models:

1. First, install Ollama from [ollama.ai](https://ollama.ai)

2. Pull the base models:
```bash
ollama pull deepseek-r1:7b
ollama pull qwen2.5:14b-instruct-q4_K_M
```

3. Create custom models with extended context windows:
```bash
# Create Deepseek model with 8k context - recommended for reasoning
ollama create deepseek-r1:7b-8k -f ollama_models/Deepseek-r1-7b-8k

# Create Qwen model with 8k context - recommended for conversation
ollama create qwen2.5:14b-instruct-8k -f ollama_models/Qwen-14b-Instruct-8k
```

Feel free to experiment with other models or context window sizes by modifying the model files in the `ollama_models` directory.

## Usage

1. Place your PDF documents in the `data` directory:
```bash
mkdir data
# Copy your PDFs into the data directory
```

2. Ingest the PDFs to create the vector database:
```bash
python ingest_pdfs.py
```

3. Run the RAG application:
```bash
python r1_smolagent_rag.py
```

This will launch a Gradio web interface where you can ask questions about your documents.

## How It Works

1. **Document Ingestion** (`ingest_pdfs.py`):
   - Loads PDFs from the `data` directory
   - Splits documents into chunks of 1000 characters with 200 character overlap
   - Creates embeddings using `sentence-transformers/all-mpnet-base-v2`
   - Stores vectors in a Chroma database

2. **RAG System** (`r1_smolagent_rag.py`):
   - Uses two LLMs: one for reasoning and one for tool calling
   - Retrieves relevant document chunks based on user queries
   - Generates responses using the retrieved context
   - Provides a Gradio web interface for interaction

## Model Selection

### HuggingFace Models
- Recommended for cloud-based inference
- Requires API token for better rate limits
- Supports a wide range of models
- Better for production use with stable API

### Ollama Models
- Recommended for local inference
- No API token required
- Runs entirely on your machine
- Better for development and testing
- Supports custom model configurations
- Lower latency but requires more system resources

## Notes
- The vector store is persisted in the `chroma_db` directory
- Default chunk size is 1000 characters with 200 character overlap
- Embeddings are generated using the `all-mpnet-base-v2` model
- The system uses a maximum of 3 relevant chunks for context
