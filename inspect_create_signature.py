#!/usr/bin/env python3
"""
Inspect the signature of GeminiEmbedder.create method.
"""

import os
import asyncio
import logging
import inspect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def inspect_create_method():
    """Inspect the create method signature."""
    
    # Import graph utils
    from agent.graph_utils import GraphitiClient
    
    # Create a Graphiti client
    client = GraphitiClient()
    
    # Initialize the client
    await client.initialize()
    
    try:
        if client.graphiti and hasattr(client.graphiti, 'embedder'):
            embedder = client.graphiti.embedder
            
            if hasattr(embedder, 'create'):
                create_method = embedder.create
                
                # Get the signature
                sig = inspect.signature(create_method)
                logger.info(f"create method signature: {sig}")
                
                # Get parameter details
                logger.info("\nParameter details:")
                for name, param in sig.parameters.items():
                    logger.info(f"  - {name}: {param}")
                
                # Try to call it with different parameter styles
                test_text = "Test embedding"
                
                # Try with just text
                try:
                    logger.info(f"\nTrying create('{test_text}')...")
                    embedding = await embedder.create(test_text)
                    logger.info(f"✅ Success! Returned embedding with dimension: {len(embedding)}")
                except Exception as e:
                    logger.error(f"❌ Failed with just text: {e}")
                
                # Try with input_data parameter
                try:
                    logger.info(f"\nTrying create(input_data='{test_text}')...")
                    embedding = await embedder.create(input_data=test_text)
                    logger.info(f"✅ Success! Returned embedding with dimension: {len(embedding)}")
                except Exception as e:
                    logger.error(f"❌ Failed with input_data: {e}")
                    
    finally:
        await client.close()


async def main():
    """Run the inspection."""
    logger.info("=" * 60)
    logger.info("Inspecting GeminiEmbedder.create Signature")
    logger.info("=" * 60)
    
    await inspect_create_method()
    
    logger.info("\n" + "=" * 60)
    logger.info("Inspection Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())