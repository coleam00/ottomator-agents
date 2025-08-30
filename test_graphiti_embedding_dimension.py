#!/usr/bin/env python3
"""
Test script to verify Graphiti embedding dimension configuration.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_graphiti_embedder():
    """Test that Graphiti is configured with the correct embedding dimension."""
    
    # Import graph utils
    from agent.graph_utils import GraphitiClient
    
    # Create a Graphiti client
    client = GraphitiClient()
    
    # Check configuration before initialization
    logger.info(f"Configured embedding dimensions: {client.embedding_dimensions}")
    logger.info(f"Embedding provider: {client.embedding_provider}")
    logger.info(f"Embedding model: {client.embedding_model}")
    
    # Initialize the client
    await client.initialize()
    
    # Check if Graphiti's embedder is properly configured
    if client.graphiti and hasattr(client.graphiti, 'embedder'):
        embedder = client.graphiti.embedder
        logger.info(f"Graphiti embedder type: {type(embedder).__name__}")
        
        # Check if the embedder has the expected configuration
        if hasattr(embedder, 'config'):
            config = embedder.config
            logger.info(f"Embedder config: {config}")
            
            # Check if embedding_dim is set correctly
            if hasattr(config, 'embedding_dim'):
                logger.info(f"✅ Embedder configured with dimension: {config.embedding_dim}")
                if config.embedding_dim != 768:
                    logger.error(f"❌ DIMENSION MISMATCH: Expected 768, got {config.embedding_dim}")
            else:
                logger.error("❌ Embedder config missing embedding_dim attribute")
        
        # Try to generate an embedding to see what dimension it produces
        if hasattr(embedder, 'embed'):
            try:
                test_text = "This is a test embedding"
                logger.info(f"Generating test embedding for: '{test_text}'")
                
                # Generate embedding
                embedding = await embedder.embed(test_text)
                
                logger.info(f"Generated embedding dimension: {len(embedding)}")
                
                if len(embedding) == 768:
                    logger.info("✅ Embedding dimension is correct (768)")
                else:
                    logger.error(f"❌ DIMENSION MISMATCH: Expected 768, got {len(embedding)}")
                    logger.error("This is the root cause of the vector dimension mismatch!")
                    
            except Exception as e:
                logger.error(f"Failed to generate test embedding: {e}")
    
    # Close the client
    await client.close()


async def test_direct_embedding():
    """Test embedding generation directly from tools."""
    from agent.tools import generate_embedding
    
    test_text = "Test embedding from tools module"
    logger.info(f"Testing direct embedding generation for: '{test_text}'")
    
    embedding = await generate_embedding(test_text)
    logger.info(f"Direct embedding dimension: {len(embedding)}")
    
    if len(embedding) == 768:
        logger.info("✅ Direct embedding dimension is correct (768)")
    else:
        logger.error(f"❌ Direct embedding dimension mismatch: {len(embedding)}")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing Graphiti Embedding Configuration")
    logger.info("=" * 60)
    
    # Test Graphiti embedder
    logger.info("\n1. Testing Graphiti embedder configuration:")
    await test_graphiti_embedder()
    
    # Test direct embedding
    logger.info("\n2. Testing direct embedding generation:")
    await test_direct_embedding()
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())