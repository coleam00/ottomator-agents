#!/usr/bin/env python3
"""
Test script to reproduce the Graphiti search dimension error.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_graphiti_search():
    """Test Graphiti search to reproduce the dimension error."""
    
    # Import graph utils
    from agent.graph_utils import GraphitiClient
    
    # Create a Graphiti client
    client = GraphitiClient()
    
    logger.info("Initializing Graphiti client...")
    await client.initialize()
    
    try:
        # Try to perform a search - this should trigger the dimension error
        logger.info("Performing search query...")
        test_query = "What is myelofibrosis?"
        
        results = await client.search(test_query)
        
        logger.info(f"Search completed successfully with {len(results)} results")
        for i, result in enumerate(results[:3]):
            logger.info(f"  Result {i+1}: {result['fact'][:100]}...")
            
    except Exception as e:
        logger.error(f"❌ Search failed with error: {e}")
        logger.error("This is likely the dimension mismatch error we're looking for")
        
        # Check if it's the dimension error
        if "vector.similarity.cosine" in str(e) or "dimension" in str(e).lower():
            logger.error("CONFIRMED: This is the vector dimension mismatch error!")
            logger.info("\nDebug Information:")
            logger.info(f"  - Embedding provider: {client.embedding_provider}")
            logger.info(f"  - Embedding model: {client.embedding_model}")
            logger.info(f"  - Configured dimensions: {client.embedding_dimensions}")
    
    finally:
        await client.close()


async def test_vector_search():
    """Test vector search in Supabase."""
    from agent.tools import vector_search_tool, VectorSearchInput
    
    logger.info("\nTesting vector search in Supabase...")
    
    try:
        query = "What is myelofibrosis?"
        results = await vector_search_tool(VectorSearchInput(query=query, limit=5))
        
        logger.info(f"✅ Vector search successful with {len(results)} results")
        for i, result in enumerate(results[:2]):
            logger.info(f"  Result {i+1}: {result.content[:100]}...")
            
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        
        # Check if it's a dimension error
        if "dimension" in str(e).lower() or "768" in str(e) or "3072" in str(e):
            logger.error("CONFIRMED: Vector dimension mismatch in Supabase!")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing for Vector Dimension Mismatch Errors")
    logger.info("=" * 60)
    
    # Test Graphiti search
    logger.info("\n1. Testing Graphiti search (Neo4j):")
    await test_graphiti_search()
    
    # Test vector search
    logger.info("\n2. Testing vector search (Supabase):")
    await test_vector_search()
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())