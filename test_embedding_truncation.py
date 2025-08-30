#!/usr/bin/env python
"""Test embedding truncation to verify it's working correctly."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.DEBUG)

async def test_embedding():
    """Test embedding generation and truncation."""
    from ingestion.embedder import EmbeddingGenerator
    from ingestion.embedding_truncator import normalize_embedding_dimension, get_target_dimension
    
    # Create embedder
    embedder = EmbeddingGenerator()
    
    # Test text
    test_text = "This is a test of embedding generation and truncation."
    
    print(f"Model: {embedder.model}")
    print(f"Target dimension: {get_target_dimension()}")
    
    # Generate single embedding
    print("\nGenerating single embedding...")
    embedding = await embedder.embed_query(test_text)
    
    print(f"Raw embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    print("\nGenerating batch embeddings...")
    batch_texts = [test_text, "Another test text", "Third test text"]
    batch_embeddings = await embedder.generate_embeddings_batch(batch_texts)
    
    for i, emb in enumerate(batch_embeddings):
        print(f"Batch embedding {i} dimension: {len(emb)}")
    
    # Test manual truncation
    print("\nTesting manual truncation...")
    truncated = normalize_embedding_dimension(embedding, 768)
    print(f"Truncated dimension: {len(truncated)}")
    print(f"First 5 values after truncation: {truncated[:5]}")

if __name__ == "__main__":
    asyncio.run(test_embedding())