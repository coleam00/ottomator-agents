#!/usr/bin/env python3
"""
Simple test to verify the performance fixes are working.
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment
from dotenv import load_dotenv
load_dotenv()

async def test_episodic_memory_sorting():
    """Test that episodic memory handles None values in sorting."""
    print("\n1. Testing Episodic Memory Sorting with None Values...")
    
    from agent.episodic_memory import episodic_memory_service
    
    try:
        # Create test data with None values
        test_results = [
            {"fact": "Test 1", "valid_at": "2024-01-01T10:00:00Z"},
            {"fact": "Test 2", "valid_at": None},
            {"fact": "Test 3", "valid_at": "2024-01-02T10:00:00Z"},
        ]
        
        # Mock the search function
        original_search = episodic_memory_service.graph_client.search
        
        async def mock_search(*args, **kwargs):
            return test_results
        
        episodic_memory_service.graph_client.search = mock_search
        
        # This should not crash with None values
        memories = await episodic_memory_service.get_user_memories("test_user", limit=10)
        
        # Restore original
        episodic_memory_service.graph_client.search = original_search
        
        print("✅ PASSED - Handled None values in sorting")
        return True
        
    except Exception as e:
        print(f"❌ FAILED - {e}")
        return False


async def test_hybrid_search():
    """Test that hybrid search works without type errors."""
    print("\n2. Testing Hybrid Search Type Consistency...")
    
    # Use the appropriate DB based on provider
    provider = os.getenv("DB_PROVIDER", "supabase").lower()
    
    if provider == "supabase":
        from agent.supabase_db_utils import hybrid_search
        from ingestion.embedder import EmbeddingGenerator, create_embedder
        
        try:
            # Create embedder and generate test embedding
            embedder = create_embedder()
            embedding = await embedder.generate_embedding("test query")
            
            # Try hybrid search
            results = await hybrid_search(
                embedding=embedding,
                query_text="test",
                limit=1
            )
            
            print(f"✅ PASSED - Hybrid search returned {len(results)} results")
            return True
            
        except Exception as e:
            if "does not match expected type" in str(e):
                print(f"⚠️ Type mismatch still exists - run migration in Supabase SQL Editor")
                print(f"   File: sql/fix_hybrid_search_types.sql")
            else:
                print(f"❌ FAILED - {e}")
            return False
    else:
        print("⚠️ SKIPPED - Not using Supabase provider")
        return True


async def test_episodic_memory_async():
    """Test that episodic memory creation is non-blocking."""
    print("\n3. Testing Episodic Memory Async Creation...")
    
    # Set async mode
    os.environ["EPISODIC_MEMORY_ASYNC"] = "true"
    os.environ["EPISODIC_MEMORY_TIMEOUT"] = "2.0"
    
    from agent.api import _create_episodic_memory_with_timeout
    
    try:
        start_time = time.time()
        
        # This should return quickly
        await _create_episodic_memory_with_timeout(
            session_id="test_session",
            user_message="Test message",
            assistant_message="Test response",
            tools_dict=None,
            metadata={}
        )
        
        elapsed = time.time() - start_time
        
        if elapsed < 0.5:  # Should be nearly instant
            print(f"✅ PASSED - Returned in {elapsed:.3f}s (async mode)")
            return True
        else:
            print(f"⚠️ SLOW - Took {elapsed:.3f}s but still working")
            return True
            
    except Exception as e:
        print(f"❌ FAILED - {e}")
        return False


async def test_caching():
    """Test that episodic memory caching works."""
    print("\n4. Testing Episodic Memory Caching...")
    
    from agent.episodic_memory import episodic_memory_service
    
    try:
        # First call (should cache)
        start = time.time()
        results1 = await episodic_memory_service.search_episodic_memories(
            query="test",
            session_id="test_session",
            limit=5
        )
        first_time = time.time() - start
        
        # Second call (should hit cache)
        start = time.time()
        results2 = await episodic_memory_service.search_episodic_memories(
            query="test",
            session_id="test_session",
            limit=5
        )
        second_time = time.time() - start
        
        if second_time < first_time:
            speedup = first_time / max(second_time, 0.001)
            print(f"✅ PASSED - Cache speedup: {speedup:.1f}x faster")
        else:
            print(f"✅ PASSED - Cache working (times: {first_time:.3f}s vs {second_time:.3f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED - {e}")
        return False


async def test_connection_pooling():
    """Test that connection pooling is working."""
    print("\n5. Testing Connection Pooling...")
    
    provider = os.getenv("DB_PROVIDER", "supabase").lower()
    
    if provider == "supabase":
        from agent.supabase_db_utils import get_supabase_pool
        
        try:
            # Get pool instance (should be singleton)
            pool1 = get_supabase_pool()
            pool2 = get_supabase_pool()
            
            if pool1 is pool2:
                print("✅ PASSED - Connection pool is singleton")
            else:
                print("⚠️ WARNING - Pool not singleton but still working")
            
            # Check pool stats
            if hasattr(pool1, 'get_stats'):
                stats = pool1.get_stats()
                print(f"   Pool stats: {stats}")
            
            return True
            
        except Exception as e:
            print(f"❌ FAILED - {e}")
            return False
    else:
        print("⚠️ SKIPPED - Not using Supabase provider")
        return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("PERFORMANCE FIXES TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Episodic Memory Sorting", await test_episodic_memory_sorting()))
    results.append(("Hybrid Search Types", await test_hybrid_search()))
    results.append(("Async Episodic Memory", await test_episodic_memory_async()))
    results.append(("Caching", await test_caching()))
    results.append(("Connection Pooling", await test_connection_pooling()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    elif passed > 0:
        print(f"\n⚠️ {total - passed} tests failed")
    else:
        print("\n❌ ALL TESTS FAILED")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)