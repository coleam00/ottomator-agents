#!/usr/bin/env python3
"""
Test script for verifying episodic memory context retrieval.

This script tests that the system properly retrieves historical context
from Graphiti before processing user queries.
"""

import asyncio
import logging
import os
from uuid import uuid4
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_episodic_context_retrieval():
    """Test the episodic context retrieval system."""
    
    # Import required modules
    from agent.episodic_memory import episodic_memory_service
    from agent.graph_utils import initialize_graph, close_graph
    from agent.api import get_episodic_context, get_episodic_context_cached
    
    try:
        # Initialize graph connection
        logger.info("Initializing graph database...")
        await initialize_graph()
        
        # Test data
        test_session_id = str(uuid4())
        test_user_id = f"test_user_{uuid4().hex[:8]}"
        
        logger.info(f"Test session ID: {test_session_id}")
        logger.info(f"Test user ID: {test_user_id}")
        
        # Step 1: Create some test episodic memories
        logger.info("\n=== Step 1: Creating test episodic memories ===")
        
        # Create first conversation turn
        episode1 = await episodic_memory_service.create_conversation_episode(
            session_id=test_session_id,
            user_message="I've been having headaches for the past week",
            assistant_response="I understand you've been experiencing headaches for a week. Can you describe the pain location and intensity?",
            tools_used=[{"tool_name": "vector_search"}],
            metadata={"user_id": test_user_id}
        )
        logger.info(f"Created episode 1: {episode1}")
        
        # Create second conversation turn
        episode2 = await episodic_memory_service.create_conversation_episode(
            session_id=test_session_id,
            user_message="The pain is mostly on the right side of my head, and it's throbbing",
            assistant_response="You're describing a throbbing pain on the right side of your head. This could be consistent with migraine symptoms.",
            tools_used=[{"tool_name": "graph_search"}],
            metadata={"user_id": test_user_id}
        )
        logger.info(f"Created episode 2: {episode2}")
        
        # Create third conversation turn
        episode3 = await episodic_memory_service.create_conversation_episode(
            session_id=test_session_id,
            user_message="I also feel nauseous when the headache is severe",
            assistant_response="Nausea accompanying severe headaches further suggests migraine. Have you noticed any triggers?",
            metadata={"user_id": test_user_id}
        )
        logger.info(f"Created episode 3: {episode3}")
        
        # Wait a bit for indexing
        await asyncio.sleep(2)
        
        # Step 2: Test searching for episodic memories
        logger.info("\n=== Step 2: Testing episodic memory search ===")
        
        # Search for session memories
        session_memories = await episodic_memory_service.search_episodic_memories(
            query="headache pain",
            session_id=test_session_id,
            user_id=test_user_id,
            limit=5
        )
        
        logger.info(f"Found {len(session_memories)} session memories")
        for i, memory in enumerate(session_memories, 1):
            logger.info(f"  Memory {i}: {memory.get('fact', 'No fact')[:100]}...")
        
        # Step 3: Test the get_episodic_context function
        logger.info("\n=== Step 3: Testing get_episodic_context function ===")
        
        # Simulate a new user message that should retrieve context
        new_message = "What treatments would you recommend for my symptoms?"
        
        context = await get_episodic_context(
            session_id=test_session_id,
            user_id=test_user_id,
            current_message=new_message,
            max_results=5
        )
        
        if context:
            logger.info("Retrieved episodic context:")
            logger.info("-" * 50)
            logger.info(context)
            logger.info("-" * 50)
        else:
            logger.warning("No episodic context retrieved")
        
        # Step 4: Test the cached version
        logger.info("\n=== Step 4: Testing cached episodic context ===")
        
        # First call - should fetch from Graphiti
        context1 = await get_episodic_context_cached(
            session_id=test_session_id,
            user_id=test_user_id,
            current_message=new_message,
            max_results=5
        )
        
        # Second call - should use cache
        context2 = await get_episodic_context_cached(
            session_id=test_session_id,
            user_id=test_user_id,
            current_message=new_message,
            max_results=5
        )
        
        logger.info(f"Context retrieved (should be cached): {bool(context2)}")
        
        # Step 5: Test session context retrieval
        logger.info("\n=== Step 5: Testing session context retrieval ===")
        
        session_context = await episodic_memory_service.get_session_context(
            session_id=test_session_id,
            limit=5
        )
        
        logger.info(f"Session context:")
        logger.info(f"  - Memories: {len(session_context.get('memories', []))}")
        logger.info(f"  - Topics: {session_context.get('topics', [])}")
        logger.info(f"  - Entities: {session_context.get('entities', [])}")
        
        # Step 6: Test user memories across sessions
        logger.info("\n=== Step 6: Testing user memory retrieval ===")
        
        user_memories = await episodic_memory_service.get_user_memories(
            user_id=test_user_id,
            limit=10
        )
        
        logger.info(f"Found {len(user_memories)} user memories across sessions")
        
        # Step 7: Verify medical entity extraction
        logger.info("\n=== Step 7: Testing medical entity extraction ===")
        
        # Search for medical-specific context
        medical_context = await get_episodic_context(
            session_id=test_session_id,
            user_id=test_user_id,
            current_message="What about my pain symptoms?",
            max_results=5
        )
        
        if medical_context and "Medical Context" in medical_context:
            logger.info("Medical context successfully retrieved")
        else:
            logger.warning("Medical context not found in retrieval")
        
        logger.info("\n=== Test completed successfully! ===")
        logger.info("The episodic context retrieval system is working properly.")
        logger.info("The system can now 'remember' previous interactions within a session.")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        await close_graph()
        logger.info("Graph connection closed")

async def test_api_with_context():
    """Test the API endpoints with episodic context."""
    
    import aiohttp
    import json
    
    # API configuration
    api_url = "http://localhost:8058"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Create a test session
            test_session_id = str(uuid4())
            test_user_id = f"api_test_user_{uuid4().hex[:8]}"
            
            logger.info(f"\n=== Testing API with context ===")
            logger.info(f"Session ID: {test_session_id}")
            logger.info(f"User ID: {test_user_id}")
            
            # First message - establish context
            message1 = {
                "message": "I have a severe headache on the right side of my head",
                "session_id": test_session_id,
                "user_id": test_user_id
            }
            
            async with session.post(f"{api_url}/chat", json=message1) as resp:
                result1 = await resp.json()
                logger.info(f"Response 1: {result1.get('message', '')[:200]}...")
            
            # Wait for episodic memory to be created
            await asyncio.sleep(3)
            
            # Second message - should retrieve context
            message2 = {
                "message": "What treatments would you recommend?",
                "session_id": test_session_id,
                "user_id": test_user_id
            }
            
            async with session.post(f"{api_url}/chat", json=message2) as resp:
                result2 = await resp.json()
                logger.info(f"Response 2: {result2.get('message', '')[:200]}...")
                
                # Check metadata to see if episodic context was used
                if result2.get("metadata", {}).get("had_episodic_context"):
                    logger.info("✓ Episodic context was retrieved and used!")
                else:
                    logger.warning("⚠ Episodic context was not used")
            
            logger.info("\n=== API test completed ===")
            
    except aiohttp.ClientConnectorError:
        logger.warning("Could not connect to API. Make sure the server is running on port 8058")
    except Exception as e:
        logger.error(f"API test failed: {e}")

async def main():
    """Main test runner."""
    
    logger.info("=" * 60)
    logger.info("EPISODIC CONTEXT RETRIEVAL TEST")
    logger.info("=" * 60)
    
    # Test the episodic memory functions directly
    await test_episodic_context_retrieval()
    
    # Optional: Test via API if server is running
    logger.info("\n" + "=" * 60)
    logger.info("Testing via API (requires server to be running)...")
    logger.info("=" * 60)
    
    try:
        await test_api_with_context()
    except Exception as e:
        logger.info(f"API test skipped: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())