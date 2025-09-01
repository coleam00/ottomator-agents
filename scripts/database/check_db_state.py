#!/usr/bin/env python
"""Check current database state and embedding dimensions."""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

async def check_database():
    """Check database state."""
    from agent.supabase_db_utils import supabase_pool
    
    supabase = supabase_pool.initialize()
    
    # Get counts
    doc_result = supabase.table("documents").select("*").execute()
    chunk_result = supabase.table("chunks").select("*").limit(5).execute()
    
    print("=" * 60)
    print("DATABASE STATE CHECK")
    print("=" * 60)
    
    # Document info
    print(f"\nDocuments in database: {len(doc_result.data)}")
    if doc_result.data:
        print("\nDocument list:")
        for doc in doc_result.data:
            created = doc.get('created_at', 'unknown')
            print(f"  - {doc['title'][:50]} (created: {created})")
    
    # Chunk info
    print(f"\nChecking first 5 chunks:")
    if chunk_result.data:
        for i, chunk in enumerate(chunk_result.data):
            embedding = chunk.get('embedding')
            if embedding:
                dim = len(embedding)
                print(f"  Chunk {i}: embedding dimension = {dim}")
                print(f"    Created: {chunk.get('created_at', 'unknown')}")
            else:
                print(f"  Chunk {i}: NO EMBEDDING")
    
    # Get unique embedding dimensions
    print("\nChecking all embedding dimensions...")
    all_chunks = supabase.table("chunks").select("embedding").execute()
    
    dimensions = set()
    for chunk in all_chunks.data:
        if chunk.get('embedding'):
            dimensions.add(len(chunk['embedding']))
    
    print(f"\nUnique embedding dimensions found: {sorted(dimensions)}")
    print(f"Total chunks: {len(all_chunks.data)}")
    
    # Clean recommendation
    if dimensions and max(dimensions) > 768:
        print("\n⚠️ WARNING: Found embeddings with dimension > 768!")
        print("   This indicates old data from before the truncation fix.")
        print("   Recommendation: Clean the database and re-ingest.")
    elif dimensions and 768 in dimensions:
        print("\n✅ Found 768-dimension embeddings (correct)")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(check_database())