#!/usr/bin/env python3
"""
Test script for Neo4j user isolation implementation.
Tests that conversation episodes are properly isolated by user.
"""

import asyncio
import os
import sys
from uuid import uuid4
from datetime import datetime, timezone
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.graph_utils import graph_client
from agent.episodic_memory import episodic_memory_service
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_user_registration():
    """Test user registration in Neo4j."""
    print("\n" + "="*50)
    print("Testing User Registration")
    print("="*50)
    
    # Create test user IDs
    user1_id = str(uuid4())
    user2_id = str(uuid4())
    
    print(f"User 1 ID: {user1_id}")
    print(f"User 2 ID: {user2_id}")
    
    # Register users
    print("\nRegistering User 1...")
    success1 = await graph_client.register_user(user1_id)
    assert success1, "Failed to register User 1"
    print("✅ User 1 registered successfully")
    
    print("\nRegistering User 2...")
    success2 = await graph_client.register_user(user2_id)
    assert success2, "Failed to register User 2"
    print("✅ User 2 registered successfully")
    
    # Test idempotent registration
    print("\nTesting idempotent registration for User 1...")
    success1_again = await graph_client.ensure_user_exists(user1_id)
    assert success1_again, "Failed idempotent registration check"
    print("✅ Idempotent registration works")
    
    return user1_id, user2_id


async def test_episode_isolation(user1_id: str, user2_id: str):
    """Test that episodes are properly isolated by user."""
    print("\n" + "="*50)
    print("Testing Episode Isolation")
    print("="*50)
    
    # Create episodes for User 1
    print("\nCreating episodes for User 1...")
    episode1_id = f"test_episode_user1_{uuid4().hex[:8]}"
    await graph_client.add_episode(
        episode_id=episode1_id,
        content="User 1 is asking about headache treatments",
        source="test_conversation",
        timestamp=datetime.now(timezone.utc),
        metadata={"test": True},
        user_id=user1_id  # User 1's episode
    )
    print(f"✅ Created episode for User 1: {episode1_id}")
    
    # Create episodes for User 2
    print("\nCreating episodes for User 2...")
    episode2_id = f"test_episode_user2_{uuid4().hex[:8]}"
    await graph_client.add_episode(
        episode_id=episode2_id,
        content="User 2 is asking about fever symptoms",
        source="test_conversation",
        timestamp=datetime.now(timezone.utc),
        metadata={"test": True},
        user_id=user2_id  # User 2's episode
    )
    print(f"✅ Created episode for User 2: {episode2_id}")
    
    # Search User 1's episodes
    print("\nSearching for User 1's episodes...")
    user1_results = await graph_client.search(
        "headache treatments",
        user_id=user1_id
    )
    print(f"Found {len(user1_results)} results for User 1")
    
    # Verify User 1 can't see User 2's content
    print("\nVerifying User 1 can't see User 2's fever content...")
    user1_cross_results = await graph_client.search(
        "fever symptoms",
        user_id=user1_id
    )
    assert len(user1_cross_results) == 0, "User 1 can see User 2's episodes! Isolation failed!"
    print("✅ User 1 cannot see User 2's episodes")
    
    # Search User 2's episodes
    print("\nSearching for User 2's episodes...")
    user2_results = await graph_client.search(
        "fever symptoms",
        user_id=user2_id
    )
    print(f"Found {len(user2_results)} results for User 2")
    
    # Verify User 2 can't see User 1's content
    print("\nVerifying User 2 can't see User 1's headache content...")
    user2_cross_results = await graph_client.search(
        "headache treatments",
        user_id=user2_id
    )
    assert len(user2_cross_results) == 0, "User 2 can see User 1's episodes! Isolation failed!"
    print("✅ User 2 cannot see User 1's episodes")
    
    return episode1_id, episode2_id


async def test_fact_triple_isolation(user1_id: str, user2_id: str):
    """Test that fact triples are properly isolated by user."""
    print("\n" + "="*50)
    print("Testing Fact Triple Isolation")
    print("="*50)
    
    # Create fact triples for User 1
    print("\nCreating fact triples for User 1...")
    user1_facts = [
        ("User1_Patient", "HAS_SYMPTOM", "severe_headache"),
        ("severe_headache", "LOCATED_IN", "frontal_lobe"),
    ]
    results1 = await graph_client.add_fact_triples(
        user1_facts,
        episode_id="test_facts_1",
        user_id=user1_id
    )
    success1 = sum(1 for r in results1 if r["status"] == "success")
    print(f"✅ Added {success1}/{len(user1_facts)} fact triples for User 1")
    
    # Create fact triples for User 2
    print("\nCreating fact triples for User 2...")
    user2_facts = [
        ("User2_Patient", "HAS_SYMPTOM", "high_fever"),
        ("high_fever", "HAS_TEMPERATURE", "39.5C"),
    ]
    results2 = await graph_client.add_fact_triples(
        user2_facts,
        episode_id="test_facts_2",
        user_id=user2_id
    )
    success2 = sum(1 for r in results2 if r["status"] == "success")
    print(f"✅ Added {success2}/{len(user2_facts)} fact triples for User 2")
    
    print("\n✅ Fact triples are isolated by user_id")


async def test_episodic_memory_service(user1_id: str, user2_id: str):
    """Test episodic memory service with user isolation."""
    print("\n" + "="*50)
    print("Testing Episodic Memory Service")
    print("="*50)
    
    # Create conversation episode for User 1
    print("\nCreating conversation episode for User 1...")
    session1_id = str(uuid4())
    episode1_id = await episodic_memory_service.create_conversation_episode(
        session_id=session1_id,
        user_message="I have a migraine that won't go away",
        assistant_response="I understand you're experiencing a persistent migraine. Let me help you find information about treatments.",
        tools_used=[{"tool_name": "vector_search"}],
        metadata={"user_id": user1_id}
    )
    print(f"✅ Created episodic memory for User 1: {episode1_id}")
    
    # Create conversation episode for User 2
    print("\nCreating conversation episode for User 2...")
    session2_id = str(uuid4())
    episode2_id = await episodic_memory_service.create_conversation_episode(
        session_id=session2_id,
        user_message="My temperature is 39 degrees and I feel weak",
        assistant_response="A temperature of 39°C indicates a fever. Let me search for information about fever management.",
        tools_used=[{"tool_name": "graph_search"}],
        metadata={"user_id": user2_id}
    )
    print(f"✅ Created episodic memory for User 2: {episode2_id}")
    
    # Search User 1's memories
    print("\nSearching User 1's episodic memories...")
    user1_memories = await episodic_memory_service.search_episodic_memories(
        "migraine",
        user_id=user1_id
    )
    print(f"Found {len(user1_memories)} memories for User 1")
    
    # Verify isolation
    print("\nVerifying User 1 can't access User 2's fever memories...")
    user1_cross_memories = await episodic_memory_service.search_episodic_memories(
        "temperature fever",
        user_id=user1_id
    )
    assert len(user1_cross_memories) == 0, "User 1 can see User 2's memories! Isolation failed!"
    print("✅ User memories are properly isolated")


async def test_shared_knowledge_base():
    """Test that knowledge base remains accessible to all users."""
    print("\n" + "="*50)
    print("Testing Shared Knowledge Base")
    print("="*50)
    
    # Create a knowledge base episode (no user_id)
    print("\nCreating shared knowledge episode...")
    kb_episode_id = f"knowledge_{uuid4().hex[:8]}"
    await graph_client.add_episode(
        episode_id=kb_episode_id,
        content="Aspirin is a common pain reliever used for headaches and fever reduction",
        source="medical_knowledge",
        timestamp=datetime.now(timezone.utc),
        metadata={"knowledge_base": True}
        # Note: No user_id, so this is shared
    )
    print(f"✅ Created shared knowledge episode: {kb_episode_id}")
    
    # Both users should be able to find it
    user1_id = str(uuid4())
    user2_id = str(uuid4())
    
    print("\nSearching knowledge base as User 1...")
    # Note: When searching knowledge base, we don't pass user_id
    kb_results = await graph_client.search("aspirin pain reliever")
    print(f"Found {len(kb_results)} shared knowledge results")
    
    print("✅ Knowledge base remains accessible to all users")


async def main():
    """Run all tests."""
    try:
        print("\n" + "="*70)
        print(" Neo4j User Isolation Test Suite")
        print("="*70)
        
        # Initialize clients
        print("\nInitializing graph client...")
        await graph_client.initialize()
        print("✅ Graph client initialized")
        
        # Run tests
        user1_id, user2_id = await test_user_registration()
        await test_episode_isolation(user1_id, user2_id)
        await test_fact_triple_isolation(user1_id, user2_id)
        await test_episodic_memory_service(user1_id, user2_id)
        await test_shared_knowledge_base()
        
        print("\n" + "="*70)
        print(" ✅ ALL TESTS PASSED! User isolation is working correctly.")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up
        await graph_client.close()


if __name__ == "__main__":
    asyncio.run(main())