#!/usr/bin/env python3
"""
Inspect the GeminiEmbedder to understand its methods.
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


async def inspect_embedder():
    """Inspect the GeminiEmbedder class and its methods."""
    
    # Import graph utils
    from agent.graph_utils import GraphitiClient
    
    # Create a Graphiti client
    client = GraphitiClient()
    
    # Initialize the client
    await client.initialize()
    
    try:
        if client.graphiti and hasattr(client.graphiti, 'embedder'):
            embedder = client.graphiti.embedder
            
            logger.info(f"Embedder type: {type(embedder).__name__}")
            logger.info(f"Embedder module: {type(embedder).__module__}")
            
            # List all methods and attributes
            logger.info("\nEmbedder methods and attributes:")
            for attr in dir(embedder):
                if not attr.startswith('_'):
                    logger.info(f"  - {attr}: {type(getattr(embedder, attr))}")
            
            # Check specific methods
            if hasattr(embedder, 'embed'):
                logger.info("\n✅ Has 'embed' method")
            else:
                logger.info("\n❌ No 'embed' method")
            
            if hasattr(embedder, 'embed_query'):
                logger.info("✅ Has 'embed_query' method")
            else:
                logger.info("❌ No 'embed_query' method")
                
            if hasattr(embedder, 'embed_text'):
                logger.info("✅ Has 'embed_text' method")
            else:
                logger.info("❌ No 'embed_text' method")
            
            # Try to call available embedding methods
            test_text = "Test embedding"
            
            if hasattr(embedder, 'embed_query'):
                try:
                    embedding = await embedder.embed_query(test_text)
                    logger.info(f"\n'embed_query' returned embedding with dimension: {len(embedding)}")
                except Exception as e:
                    logger.error(f"'embed_query' failed: {e}")
            
            if hasattr(embedder, 'embed_text'):
                try:
                    embedding = await embedder.embed_text(test_text)
                    logger.info(f"'embed_text' returned embedding with dimension: {len(embedding)}")
                except Exception as e:
                    logger.error(f"'embed_text' failed: {e}")
                    
    finally:
        await client.close()


async def main():
    """Run the inspection."""
    logger.info("=" * 60)
    logger.info("Inspecting GeminiEmbedder")
    logger.info("=" * 60)
    
    await inspect_embedder()
    
    logger.info("\n" + "=" * 60)
    logger.info("Inspection Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())