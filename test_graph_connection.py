#!/usr/bin/env python3
"""Test the connection to Neo4j and Graphiti setup."""

import asyncio
import os
from dotenv import load_dotenv
from agent.graph_utils import GraphitiClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    """Test Neo4j connection via Graphiti."""
    
    load_dotenv()
    
    print("=" * 60)
    print("TESTING GRAPHITI CONNECTION TO NEO4J")
    print("=" * 60)
    
    try:
        # Initialize GraphitiClient
        client = GraphitiClient()
        await client.initialize()
        print("✅ Successfully initialized Graphiti client")
        
        # Test add a simple episode
        from datetime import datetime, timezone
        test_episode_id = f"test_episode_{datetime.now().timestamp()}"
        
        print("\nTesting episode creation...")
        await client.add_episode(
            episode_id=test_episode_id,
            content="This is a test episode to verify Neo4j connectivity.",
            source="Connection Test",
            timestamp=datetime.now(timezone.utc),
            metadata={"test": True},
            group_id="0"
        )
        print("✅ Successfully added test episode to Neo4j")
        
        # Close the connection
        await client.close()
        print("✅ Connection closed successfully")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Ready to build knowledge graph")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. Neo4j credentials in .env file")
        print("2. Neo4j instance is running and accessible")
        print("3. LLM API keys are set correctly")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    exit(0 if success else 1)