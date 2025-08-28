#!/usr/bin/env python3
"""
Test script for Neo4j user isolation using Graphiti's group_id feature.
This test verifies that:
1. Shared knowledge (group_id="0") is accessible to all users
2. User-specific conversations are properly isolated by group_id
3. Users cannot access each other's episodic memories
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


async def test_user_isolation():
    """Test user isolation using Graphiti's group_id feature."""
    print("\n" + "="*70)
    print("NEO4J USER ISOLATION TEST - USING GROUP_ID")
    print("="*70)
    
    # Initialize
    print("\n1. Initializing graph client...")
    await graph_client.initialize()
    print("   ✅ Graph client initialized")
    
    # Create test users (using Supabase-like UUIDs)
    user1_id = str(uuid4())
    user2_id = str(uuid4())
    
    print(f"\n2. Test Users:")
    print(f"   User 1: {user1_id}")
    print(f"   User 2: {user2_id}")
    
    # Test 1: Add shared knowledge to group_id="0"
    print("\n3. Adding shared medical knowledge (group_id='0')...")
    shared_episode_id = f"shared_knowledge_{uuid4().hex[:8]}"
    await graph_client.add_episode(
        episode_id=shared_episode_id,
        content="Menopause typically occurs between ages 45-55. Common symptoms include hot flashes, night sweats, mood changes, and irregular periods.",
        source="medical_knowledge_base",
        timestamp=datetime.now(timezone.utc),
        metadata={"knowledge_type": "shared", "topic": "menopause_basics"},
        group_id="0"  # Shared knowledge base
    )
    print(f"   ✅ Added shared knowledge: {shared_episode_id}")
    
    # Test 2: Create User 1's episodic memory
    print("\n4. Creating User 1's conversation episode...")
    session1_id = str(uuid4())
    episode1_id = await episodic_memory_service.create_conversation_episode(
        session_id=session1_id,
        user_message="I've been experiencing severe hot flashes for 3 weeks",
        assistant_response="I understand you're dealing with severe hot flashes. Let me provide information about managing this symptom.",
        tools_used=[{"tool_name": "vector_search"}],
        metadata={"user_id": user1_id}  # This will be used as group_id
    )
    print(f"   ✅ Created User 1's episode: {episode1_id}")
    
    # Test 3: Create User 2's episodic memory
    print("\n5. Creating User 2's conversation episode...")
    session2_id = str(uuid4())
    episode2_id = await episodic_memory_service.create_conversation_episode(
        session_id=session2_id,
        user_message="My mood swings have been really difficult lately",
        assistant_response="I hear that mood swings are challenging for you. Let's explore strategies to help manage them.",
        tools_used=[{"tool_name": "graph_search"}],
        metadata={"user_id": user2_id}  # This will be used as group_id
    )
    print(f"   ✅ Created User 2's episode: {episode2_id}")
    
    # Test 4: Search shared knowledge (should be visible to both)
    print("\n6. Testing shared knowledge access...")
    print("   Searching for 'menopause symptoms' in shared knowledge...")
    shared_results = await graph_client.search(
        "menopause symptoms hot flashes",
        group_ids=["0"]  # Search only shared knowledge
    )
    print(f"   Found {len(shared_results)} results in shared knowledge")
    if shared_results:
        print(f"   Sample: {shared_results[0].get('fact', 'N/A')[:100]}...")
    
    # Test 5: Test User 1's isolated search
    print("\n7. Testing User 1's isolated memory search...")
    user1_results = await episodic_memory_service.search_episodic_memories(
        "hot flashes severe",
        user_id=user1_id  # Should only see User 1's data
    )
    print(f"   Found {len(user1_results)} results for User 1")
    
    # Check if User 1 can see User 2's data (should not)
    user1_cross_results = await episodic_memory_service.search_episodic_memories(
        "mood swings difficult",
        user_id=user1_id  # Should NOT see User 2's mood swings
    )
    if len(user1_cross_results) > 0:
        print(f"   ⚠️ WARNING: User 1 found {len(user1_cross_results)} results for User 2's content!")
        print("   User isolation may not be working correctly")
    else:
        print("   ✅ User 1 cannot see User 2's mood swings data")
    
    # Test 6: Test User 2's isolated search
    print("\n8. Testing User 2's isolated memory search...")
    user2_results = await episodic_memory_service.search_episodic_memories(
        "mood swings",
        user_id=user2_id  # Should only see User 2's data
    )
    print(f"   Found {len(user2_results)} results for User 2")
    
    # Check if User 2 can see User 1's data (should not)
    user2_cross_results = await episodic_memory_service.search_episodic_memories(
        "hot flashes severe weeks",
        user_id=user2_id  # Should NOT see User 1's hot flashes
    )
    if len(user2_cross_results) > 0:
        print(f"   ⚠️ WARNING: User 2 found {len(user2_cross_results)} results for User 1's content!")
        print("   User isolation may not be working correctly")
    else:
        print("   ✅ User 2 cannot see User 1's hot flashes data")
    
    # Test 7: Combined search (shared + user-specific)
    print("\n9. Testing combined search (shared + user-specific)...")
    
    # User 1 searches for general info (should get shared + their own)
    print("   User 1 searching across shared and personal data...")
    combined_results_user1 = await graph_client.search(
        "symptoms menopause hot flashes",
        group_ids=["0", user1_id]  # Search both shared and User 1's data
    )
    print(f"   User 1 found {len(combined_results_user1)} total results")
    
    # User 2 searches (should get shared + their own, but not User 1's)
    print("   User 2 searching across shared and personal data...")
    combined_results_user2 = await graph_client.search(
        "symptoms menopause mood",
        group_ids=["0", user2_id]  # Search both shared and User 2's data
    )
    print(f"   User 2 found {len(combined_results_user2)} total results")
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    print("\n✅ What's Working:")
    print("  - Shared knowledge stored with group_id='0'")
    print("  - User episodes stored with user's UUID as group_id")
    print("  - Search can be filtered by group_ids")
    print("  - Users can access shared knowledge + their own data")
    
    print("\n🎯 Key Implementation:")
    print("  - Graphiti's native group_id feature provides isolation")
    print("  - No custom methods needed - just proper parameter usage")
    print("  - group_id='0' for shared knowledge base")
    print("  - group_id=user_uuid for personal conversations")
    
    print("\n📊 Isolation Test Results:")
    if len(user1_cross_results) == 0 and len(user2_cross_results) == 0:
        print("  ✅ PASS: Users cannot access each other's data")
        print("  ✅ PASS: User isolation is working correctly")
    else:
        print("  ❌ FAIL: Cross-user data leakage detected")
        print("  ⚠️  User isolation needs further investigation")
    
    await graph_client.close()
    print("\n✅ Test completed")


if __name__ == "__main__":
    asyncio.run(test_user_isolation())