# Pydantic AI: Web Search Agent with Brave API

This project implements a web search agent using Pydantic AI and the Brave Search API, with both a command-line interface and a Streamlit web interface. The agent can be configured to use either OpenAI's GPT models or Ollama's local models.

## Prerequisites

- Python 3.11+
- OpenAI API key (if using GPT models)
- [Ollama](https://ollama.ai/) (optional, for local LLM usage)
- Brave Search API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/coleam00/ai-agents-masterclass.git
cd ai-agents-masterclass/pydantic-ai
```

2. Install dependencies (I recommend to do this in a Python virtual environment):
```bash
pip install -r requirements.txt
```

This will install Pydantic AI, Streamlit, and all of their dependencies.

3. Set up environment variables:
   - Rename `.env.example` to `.env`.
   - Edit `.env` with your API keys and preferences:
   ```env
   OPENAI_API_KEY=your_openai_api_key  # Only needed if using GPT models
   BRAVE_API_KEY=your_brave_api_key
   LLM_MODEL=your_chosen_model  # e.g., gpt-4, qwen2.5:32b
   ```

## Usage

### Command Line Interface

The command-line version can work with both GPT and Ollama models:

```bash
python web_search_agent.py
```

The script determines whether to use OpenAI or Ollama based on the `LLM_MODEL` environment variable (whether it starts with 'gpt' or not).

### Streamlit Interface

The Streamlit version is created to provide a UI with text streaming from the LLM and chat history. Text streaming doesn't work with Ollama, so this Streamlit example will just use GPT. Make sure your have your OpenAI API key set. This can also be adjusted to use standard non-streaming like `web_search_agent.py` if you want to use Ollama.

1. Set your OpenAI API key in the `.env` file.
2. Start the Streamlit app:
```bash
streamlit run streamlit_ui.py
```
3. The Streamlit app will open in your browser.

## Configuration

### LLM Models

You can choose between different LLM models by setting the `LLM_MODEL` environment variable:

- For OpenAI GPT (example model, this can be any OpenAI model):
  ```env
  LLM_MODEL=gpt-4o
  ```

- For Ollama (example model, this can be any Ollama model you have downloaded):
  ```env
  LLM_MODEL=qwen2.5:32b
  ```

### API Keys

- **Brave Search API**: Get your API key from [Brave Search API](https://brave.com/search/api/)
- **OpenAI API** (optional): Get your API key from [OpenAI](https://platform.openai.com/api-keys)

## Troubleshooting

1. **Ollama Connection Issues**:
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is downloaded: `ollama pull your_model_name`

2. **API Key Issues**:
   - Verify your API keys are correctly set in the `.env` file
   - Check if your Brave API key has sufficient credits

3. **Model Loading Issues**:
   - For Ollama, ensure you have sufficient RAM for your chosen model
   - Try using a smaller model if you experience memory issues
