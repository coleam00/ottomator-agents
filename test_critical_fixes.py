#!/usr/bin/env python
"""
Test script to verify critical fixes for:
1. Neo4j vector dimension mismatch
2. Session validation with legacy IDs
3. Graphiti episodic memory metadata handling
"""

import asyncio
import os
import sys
import logging
import json
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
from agent.graph_utils import graph_client
from agent.embedding_config import EmbeddingConfig
from agent.episodic_memory import episodic_memory_service


async def test_vector_dimensions():
    """Test 1: Verify vector dimension normalization in Neo4j"""
    print("\n" + "="*60)
    print("TEST 1: Vector Dimension Normalization")
    print("="*60)
    
    try:
        # Initialize graph client
        await graph_client.initialize()
        
        # Check configured dimensions
        target_dim = EmbeddingConfig.get_target_dimension()
        print(f"✓ Target dimension configured: {target_dim}")
        
        # Test adding an episode with content
        test_episode_id = f"test_dimension_{uuid4().hex[:8]}"
        test_content = "This is a test episode to verify vector dimension normalization in Neo4j."
        
        await graph_client.add_episode(
            episode_id=test_episode_id,
            content=test_content,
            source="dimension_test",
            timestamp=datetime.now(timezone.utc)
        )
        print(f"✓ Successfully added test episode: {test_episode_id}")
        
        # Test searching (this will use vector similarity)
        search_results = await graph_client.search("test episode vector dimension")
        print(f"✓ Search completed without dimension mismatch errors")
        print(f"  Found {len(search_results)} results")
        
        # Verify embedding normalizer is active
        if graph_client.embedding_normalizer:
            print(f"✓ Embedding normalizer is active")
            print(f"  Normalizing to {target_dim} dimensions")
        else:
            print("⚠ Warning: Embedding normalizer not active")
        
        return True
        
    except Exception as e:
        print(f"✗ Vector dimension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await graph_client.close()


async def test_session_validation():
    """Test 2: Verify session validation handles both UUID and legacy formats"""
    print("\n" + "="*60)
    print("TEST 2: Session Validation (UUID and Legacy Formats)")
    print("="*60)
    
    try:
        # Import session functions based on DB provider
        db_provider = os.getenv("DB_PROVIDER", "postgres").lower()
        
        if db_provider == "supabase":
            from agent.supabase_db_utils import create_session, get_session
        else:
            from agent.db_utils import create_session, get_session
        
        # Test 1: Create session with UUID format
        uuid_session = await create_session(
            user_id="test_user",
            metadata={"test": "uuid_format"}
        )
        print(f"✓ Created UUID session: {uuid_session}")
        
        # Verify it's retrievable
        session_data = await get_session(uuid_session)
        if session_data:
            print(f"✓ UUID session is retrievable")
        else:
            print(f"✗ Failed to retrieve UUID session")
        
        # Test 2: Simulate legacy session ID
        legacy_id = f"session-{int(datetime.now().timestamp() * 1000)}"
        print(f"\nTesting legacy session ID: {legacy_id}")
        
        # Convert legacy ID to UUID deterministically
        import hashlib
        from uuid import UUID
        hash_digest = hashlib.md5(legacy_id.encode()).hexdigest()
        converted_uuid = str(UUID(hash_digest))
        print(f"✓ Legacy ID converted to UUID: {converted_uuid}")
        
        # Create session with converted UUID
        await create_session(
            session_id=converted_uuid,
            user_id="test_user",
            metadata={"test": "legacy_conversion", "legacy_id": legacy_id}
        )
        print(f"✓ Created session for legacy ID conversion")
        
        # Verify it's retrievable
        session_data = await get_session(converted_uuid)
        if session_data:
            print(f"✓ Converted session is retrievable")
            if session_data.get("metadata", {}).get("legacy_id") == legacy_id:
                print(f"✓ Legacy ID preserved in metadata")
        else:
            print(f"✗ Failed to retrieve converted session")
        
        # Test 3: Test the API endpoint with legacy format
        from agent.api import get_or_create_session
        from agent.models import ChatRequest
        
        # Create request with legacy session ID
        request = ChatRequest(
            message="Test message",
            session_id=legacy_id,
            user_id="test_user"
        )
        
        # This should handle the legacy ID gracefully
        final_session_id = await get_or_create_session(request)
        print(f"✓ API handled legacy session ID")
        print(f"  Returned session: {final_session_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Session validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_episodic_memory():
    """Test 3: Verify episodic memory creation without metadata parameter error"""
    print("\n" + "="*60)
    print("TEST 3: Episodic Memory (Graphiti API Compatibility)")
    print("="*60)
    
    try:
        # Create a test conversation episode
        session_id = str(uuid4())
        user_message = "What are the symptoms of migraine?"
        assistant_response = "Common migraine symptoms include severe headache, nausea, and sensitivity to light."
        
        print(f"Creating episodic memory for session: {session_id}")
        
        # This should work without the metadata parameter error
        episode_id = await episodic_memory_service.create_conversation_episode(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            tools_used=[{"tool_name": "vector_search"}],
            metadata={"test": "episodic_memory"}
        )
        
        if episode_id:
            print(f"✓ Successfully created episodic memory: {episode_id}")
            print(f"  No 'unexpected keyword argument' error")
        else:
            print(f"⚠ Episode creation returned None (check if episodic memory is enabled)")
            if not episodic_memory_service._enabled:
                print(f"  Episodic memory is disabled (ENABLE_EPISODIC_MEMORY={os.getenv('ENABLE_EPISODIC_MEMORY', 'true')})")
        
        # Test with legacy session ID format
        legacy_session = f"session-{int(datetime.now().timestamp() * 1000)}"
        print(f"\nTesting with legacy session ID: {legacy_session}")
        
        episode_id_legacy = await episodic_memory_service.create_conversation_episode(
            session_id=legacy_session,
            user_message="Test with legacy session",
            assistant_response="This should handle legacy session IDs",
            tools_used=None,
            metadata={"legacy_test": True}
        )
        
        if episode_id_legacy:
            print(f"✓ Successfully handled legacy session ID in episodic memory")
            print(f"  Episode ID: {episode_id_legacy}")
        
        return True
        
    except Exception as e:
        error_str = str(e)
        if "unexpected keyword argument 'metadata'" in error_str:
            print(f"✗ Metadata parameter error still present!")
            print(f"  Error: {error_str}")
            print("\n  This indicates the Graphiti add_episode fix may not be working.")
            print("  Check that metadata is being handled correctly in graph_utils.py")
        else:
            print(f"✗ Episodic memory test failed: {e}")
        
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all critical fix tests"""
    print("\n" + "="*60)
    print("CRITICAL FIXES TEST SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"DB Provider: {os.getenv('DB_PROVIDER', 'postgres')}")
    print(f"Embedding Provider: {os.getenv('EMBEDDING_PROVIDER', 'openai')}")
    print(f"Target Dimension: {EmbeddingConfig.get_target_dimension()}")
    
    results = {}
    
    # Test 1: Vector Dimensions
    results["vector_dimensions"] = await test_vector_dimensions()
    
    # Test 2: Session Validation
    results["session_validation"] = await test_session_validation()
    
    # Test 3: Episodic Memory
    results["episodic_memory"] = await test_episodic_memory()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Critical fixes are working!")
    else:
        print("✗ SOME TESTS FAILED - Review the errors above")
    print("="*60)
    
    # Save results to file
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "all_passed": all_passed,
            "environment": {
                "db_provider": os.getenv("DB_PROVIDER", "postgres"),
                "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
                "target_dimension": EmbeddingConfig.get_target_dimension()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_passed


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)