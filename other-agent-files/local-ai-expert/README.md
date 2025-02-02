# Local AI Expert Agent

Author: [Cole Medin](https://www.youtube.com/@ColeMedin)

This n8n-powered agent serves as your personal consultant for local AI deployments and open-source Large Language Models (LLMs). It has been prompted with general guidance for LLM hardware requirements and deployment strategies. And it also is able to search through the Ollama and HuggingFace model catalogs for information on specific models.

## Features

- Provides information about trending open-source LLMs
- Advises on hardware requirements for running different LLMs
- Offers guidance on local AI deployment strategies
- Answers questions about model optimization and performance
- Helps troubleshoot common local AI setup issues

## Technical Details

- **Platform**: n8n workflow automation
- **Model**: Anthropic Claude 3.5 Haiku
- **Integration**: Webhook-based interface
- **Response Format**: Structured JSON with HTML content

## Use Cases

1. **LLM Selection Guidance**
   - Get recommendations for open-source LLMs based on your specific needs
   - Compare different models' capabilities and requirements

2. **Hardware Requirements**
   - Understand GPU/CPU requirements for specific models
   - Get memory and storage recommendations
   - Learn about optimization techniques for your hardware

3. **Deployment Assistance**
   - Step-by-step guidance for local model deployment
   - Configuration best practices
   - Container and environment setup advice

4. **Troubleshooting**
   - Common issues and solutions
   - Performance optimization tips
   - Resource management guidance

## How to Use

1. Send a POST request to the agent's webhook endpoint
2. Include your query in the request body with the following structure:
```json
{
    "query": "Your question about local AI",
    "user_id": "your_user_id",
    "request_id": "unique_request_id",
    "session_id": "session_identifier"
}
```

## Example Queries

- "What are the current top 3 open-source LLMs for local deployment?"
- "What are the minimum hardware requirements to run Llama 3.2 locally?"
- "How can I optimize my local LLM for better performance?"
- "What's the best way to quantize a large language model?"

## Limitations

- Response times may vary based on query complexity
- Hardware recommendations are based on general guidelines and may need adjustment for specific use cases
- Model information is current as of available training data

## Contributing

This agent is part of the oTTomator agents collection. For contributions or issues, please refer to the main repository guidelines.
