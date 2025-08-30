#!/usr/bin/env python
"""Debug the embedding flow to find where truncation is failing."""

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

async def test_full_flow():
    """Test the full embedding flow."""
    
    # Import the components
    from ingestion.chunker import DocumentChunk
    from ingestion.embedder import EmbeddingGenerator
    from ingestion.embedding_truncator import get_target_dimension
    
    # Create a test chunk
    test_chunk = DocumentChunk(
        content="This is a test chunk for embedding generation.",
        index=0,
        start_char=0,
        end_char=50,
        metadata={"test": True},
        token_count=10
    )
    
    print(f"Target dimension from config: {get_target_dimension()}")
    
    # Create embedder
    embedder = EmbeddingGenerator()
    print(f"Embedder model: {embedder.model}")
    
    # Generate embeddings for the chunk
    print("\nGenerating embeddings for chunk...")
    embedded_chunks = await embedder.embed_chunks([test_chunk])
    
    if embedded_chunks:
        chunk = embedded_chunks[0]
        if hasattr(chunk, 'embedding') and chunk.embedding:
            print(f"Embedding dimension after embed_chunks: {len(chunk.embedding)}")
            print(f"First 5 values: {chunk.embedding[:5]}")
            
            # Now test what happens when we prepare for database
            chunk_dict = {
                "document_id": "test-doc-id",
                "content": chunk.content,
                "embedding": chunk.embedding if hasattr(chunk, 'embedding') and chunk.embedding else None,
                "chunk_index": chunk.index,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count
            }
            
            print(f"\nChunk dict embedding dimension: {len(chunk_dict['embedding']) if chunk_dict['embedding'] else 0}")
            
            # Test direct database insertion
            print("\nTesting database insertion...")
            from agent.supabase_db_utils import supabase_pool
            
            # DON'T actually insert, just check what would be sent
            print(f"Would send embedding with dimension: {len(chunk_dict['embedding'])}")
        else:
            print("No embedding generated!")
    else:
        print("No chunks returned!")

if __name__ == "__main__":
    asyncio.run(test_full_flow())