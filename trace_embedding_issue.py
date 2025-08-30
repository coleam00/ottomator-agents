#!/usr/bin/env python
"""Trace the exact issue with embedding dimensions."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_openai_client_with_gemini():
    """Test how the OpenAI client handles Gemini embeddings."""
    
    # Get the embedding client as used in the code
    from agent.providers import get_embedding_client, get_embedding_model
    
    client = get_embedding_client()
    model = get_embedding_model()
    
    print(f"Using model: {model}")
    print(f"Client base URL: {client.base_url if hasattr(client, 'base_url') else 'N/A'}")
    
    # Generate an embedding
    test_text = "Test embedding generation"
    
    response = await client.embeddings.create(
        model=model,
        input=test_text
    )
    
    raw_embedding = response.data[0].embedding
    print(f"Raw embedding dimension from API: {len(raw_embedding)}")
    print(f"First 5 values: {raw_embedding[:5]}")
    
    # Now test with truncation
    from ingestion.embedding_truncator import normalize_embedding_dimension
    
    truncated = normalize_embedding_dimension(raw_embedding, 768)
    print(f"After truncation: {len(truncated)}")
    
    return raw_embedding

if __name__ == "__main__":
    asyncio.run(test_openai_client_with_gemini())