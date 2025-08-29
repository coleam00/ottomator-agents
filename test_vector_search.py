#!/usr/bin/env python3
"""
Test vector search functionality after ingestion.
"""

import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from ingestion.embedder import EmbeddingGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_vector_search():
    """Test vector search with a sample query."""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize embedding generator
        embedder = EmbeddingGenerator()
        
        # Test queries
        test_queries = [
            "What are the symptoms of menopause?",
            "How can I manage hot flashes?",
            "What is mindfulness meditation?",
            "What supplements help with menopause?"
        ]
        
        logger.info("=" * 60)
        logger.info("TESTING VECTOR SEARCH")
        logger.info("=" * 60)
        
        for query in test_queries:
            logger.info(f"\n📝 Query: {query}")
            
            # Generate embedding for query
            query_embedding = await embedder.generate_embedding(query)
            
            if not query_embedding:
                logger.error(f"Failed to generate embedding for query: {query}")
                continue
            
            # Perform vector search using RPC function
            result = supabase.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding,
                    "similarity_threshold": 0.5,
                    "match_count": 3
                }
            ).execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} relevant chunks:")
                for i, chunk in enumerate(result.data[:3], 1):
                    content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                    logger.info(f"  {i}. Similarity: {chunk['similarity']:.3f}")
                    logger.info(f"     Document: {chunk.get('document_title', 'Unknown')}")
                    logger.info(f"     Preview: {content_preview}")
            else:
                logger.warning("No results found for this query")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Vector search test complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during vector search test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vector_search())