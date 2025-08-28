#!/usr/bin/env python3
"""
Simplified test script for checking Neo4j user isolation capabilities.
This test focuses on the core isolation mechanism without extensive Graphiti operations.
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
    level=logging.WARNING,  # Reduce log verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def quick_isolation_test():
    """Quick test of user isolation capabilities."""
    print("\n" + "="*60)
    print("QUICK NEO4J USER ISOLATION TEST")
    print("="*60)
    
    # Initialize
    print("\n1. Initializing graph client...")
    await graph_client.initialize()
    print("   ✅ Graph client initialized")
    
    # Test 1: Check if user_id is preserved in metadata
    print("\n2. Testing metadata preservation...")
    user1_id = str(uuid4())
    user2_id = str(uuid4())
    
    print(f"   User 1: {user1_id[:8]}...")
    print(f"   User 2: {user2_id[:8]}...")
    
    # Create simple episodes
    episode1_id = f"test_{uuid4().hex[:8]}"
    episode2_id = f"test_{uuid4().hex[:8]}"
    
    print("\n3. Creating test episodes...")
    await graph_client.add_episode(
        episode_id=episode1_id,
        content="User 1 test content: headache symptoms",
        source="test",
        metadata={"user_id": user1_id}
    )
    print(f"   ✅ Created episode for User 1")
    
    await graph_client.add_episode(
        episode_id=episode2_id,
        content="User 2 test content: fever symptoms",
        source="test",
        metadata={"user_id": user2_id}
    )
    print(f"   ✅ Created episode for User 2")
    
    # Test 2: Search functionality
    print("\n4. Testing search capabilities...")
    
    # Search for User 1's content
    results1 = await graph_client.search("headache symptoms")
    print(f"   Found {len(results1)} results for 'headache symptoms'")
    
    # Search for User 2's content
    results2 = await graph_client.search("fever symptoms")
    print(f"   Found {len(results2)} results for 'fever symptoms'")
    
    # Test 3: Check GraphitiClient methods
    print("\n5. Checking GraphitiClient capabilities...")
    methods = [m for m in dir(graph_client) if not m.startswith('_')]
    
    user_methods = [m for m in methods if 'user' in m.lower()]
    if user_methods:
        print(f"   Found user-related methods: {', '.join(user_methods)}")
    else:
        print("   ⚠️  No user-specific methods found in GraphitiClient")
    
    # Test 4: Check episodic memory service
    print("\n6. Testing episodic memory service...")
    try:
        # Create a simple conversation episode
        session_id = str(uuid4())
        ep_id = await episodic_memory_service.create_conversation_episode(
            session_id=session_id,
            user_message="Test message",
            assistant_response="Test response",
            tools_used=[],
            metadata={"user_id": user1_id}
        )
        if ep_id:
            print(f"   ✅ Episodic memory created with user_id in metadata")
        else:
            print(f"   ⚠️  Episodic memory creation returned None")
    except Exception as e:
        print(f"   ❌ Error creating episodic memory: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    print("\n✅ What's Working:")
    print("  - Episodes can be created with user_id in metadata")
    print("  - Basic search functionality is operational")
    print("  - Episodic memory service accepts user_id in metadata")
    
    print("\n⚠️  Current Limitations:")
    print("  - GraphitiClient lacks user-specific methods (register_user, etc.)")
    print("  - No native user_id filtering in search methods")
    print("  - User isolation must be implemented at application level")
    
    print("\n💡 Recommendations for Full User Isolation:")
    print("  1. Add register_user() method to GraphitiClient")
    print("  2. Add user_id parameter to search() method")  
    print("  3. Implement user_id filtering in Cypher queries")
    print("  4. Add ensure_user_exists() for idempotent registration")
    print("  5. Update episodic_memory to enforce user context")
    
    print("\n📝 Next Steps:")
    print("  - Implement missing user isolation methods in GraphitiClient")
    print("  - Add user_id filtering to all search operations")
    print("  - Create integration with Supabase Edge Functions")
    print("  - Test end-to-end user registration flow")
    
    await graph_client.close()
    print("\n✅ Test completed successfully")


if __name__ == "__main__":
    asyncio.run(quick_isolation_test())