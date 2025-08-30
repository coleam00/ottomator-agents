#!/usr/bin/env python3
"""
Test script to verify all critical bug fixes.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestChunk:
    """Test chunk with potentially None metadata."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Any]]
    token_count: int


async def test_optimized_embedder_metadata():
    """Test that optimized_embedder handles None metadata safely."""
    from ingestion.optimized_embedder import OptimizedEmbeddingGenerator
    from ingestion.chunker import DocumentChunk
    
    logger.info("Testing optimized_embedder with None metadata...")
    
    # Create embedder
    embedder = OptimizedEmbeddingGenerator(
        batch_size=2,
        use_redis_cache=False  # Disable Redis for testing
    )
    
    # Create chunks with None metadata
    chunks = [
        DocumentChunk(
            content="Test chunk 1",
            index=0,
            start_char=0,
            end_char=12,
            metadata=None,  # None metadata
            token_count=3
        ),
        DocumentChunk(
            content="Test chunk 2",
            index=1,
            start_char=13,
            end_char=25,
            metadata={"key": "value"},  # Valid metadata
            token_count=3
        )
    ]
    
    try:
        # This should not crash with None metadata
        embedded_chunks = []
        async for chunk in embedder.embed_chunks_streaming(chunks):
            embedded_chunks.append(chunk)
            # Verify metadata was properly merged
            assert "embedding_model" in chunk.metadata
            assert "embedding_dimension" in chunk.metadata
            logger.info(f"✓ Chunk {chunk.index} embedded successfully with metadata: {chunk.metadata}")
        
        logger.info("✓ Optimized embedder metadata fix PASSED")
        return True
    except Exception as e:
        logger.error(f"✗ Optimized embedder metadata fix FAILED: {e}")
        return False
    finally:
        await embedder.close()


async def test_lru_cache_behavior():
    """Test that LRU cache properly updates access time on hits."""
    from ingestion.optimized_embedder import OptimizedEmbeddingGenerator
    import time
    
    logger.info("Testing LRU cache behavior...")
    
    # Create embedder with small cache for testing
    embedder = OptimizedEmbeddingGenerator(
        use_redis_cache=False,
        cache_ttl=3600  # Long TTL to test LRU not expiration
    )
    embedder.cache_max_size = 3  # Small cache for testing
    
    # Add cache entries
    cache_key1 = embedder._get_cache_key("text1")
    cache_key2 = embedder._get_cache_key("text2")
    cache_key3 = embedder._get_cache_key("text3")
    
    # Manually add to cache with timestamps
    embedder.memory_cache[cache_key1] = ([1.0] * 768, datetime.now())
    await asyncio.sleep(0.1)
    embedder.memory_cache[cache_key2] = ([2.0] * 768, datetime.now())
    await asyncio.sleep(0.1)
    embedder.memory_cache[cache_key3] = ([3.0] * 768, datetime.now())
    
    # Access cache_key1 to update its timestamp
    result = await embedder._get_from_cache("text1")
    assert result is not None
    
    # Now cache_key1 should have the newest timestamp
    # Add a fourth entry to trigger eviction
    cache_key4 = embedder._get_cache_key("text4")
    await embedder._save_to_cache("text4", [4.0] * 768)
    
    # cache_key2 should be evicted (oldest access time)
    assert cache_key1 in embedder.memory_cache, "Key1 should still be in cache (recently accessed)"
    assert cache_key2 not in embedder.memory_cache, "Key2 should be evicted (oldest access)"
    assert cache_key3 in embedder.memory_cache, "Key3 should still be in cache"
    assert cache_key4 in embedder.memory_cache, "Key4 should be in cache (just added)"
    
    logger.info("✓ LRU cache behavior fix PASSED")
    await embedder.close()
    return True


def test_agent_post_init():
    """Test that agent.py has single __post_init__ with proper logic."""
    from agent.agent import AgentDependencies
    
    logger.info("Testing agent __post_init__ fix...")
    
    # Test with valid UUID
    valid_uuid = str(uuid4())
    deps1 = AgentDependencies(session_id=valid_uuid)
    assert deps1.search_preferences is not None
    assert deps1.search_preferences["use_vector"] == True
    logger.info(f"✓ Valid UUID handled correctly: {valid_uuid}")
    
    # Test with invalid session_id (should log error but not crash)
    deps2 = AgentDependencies(session_id="invalid-uuid")
    assert deps2.search_preferences is not None
    logger.info("✓ Invalid UUID handled gracefully")
    
    # Test with None search_preferences
    deps3 = AgentDependencies(session_id=valid_uuid, search_preferences=None)
    assert deps3.search_preferences == {
        "use_vector": True,
        "use_graph": True,
        "default_limit": 10
    }
    logger.info("✓ Default search_preferences set correctly")
    
    logger.info("✓ Agent __post_init__ fix PASSED")
    return True


async def test_episodic_memory_session_id():
    """Test that episodic memory doesn't leak raw session_id."""
    from agent.episodic_memory import EpisodicMemoryService
    
    logger.info("Testing episodic memory session_id sanitization...")
    
    service = EpisodicMemoryService()
    service._enabled = True  # Enable for testing
    
    # Test with various session_id formats
    test_cases = [
        (str(uuid4()), "Valid UUID"),
        ("invalid-uuid-format", "Invalid format"),
        ("../../malicious/path", "Path traversal attempt"),
        ("very-long-" * 20, "Very long string")
    ]
    
    for session_id, description in test_cases:
        try:
            # Create conversation episode
            episode_id = await service.create_conversation_episode(
                session_id=session_id,
                user_message="Test message",
                assistant_response="Test response",
                metadata={"test": True}
            )
            
            # Check that episode_id doesn't contain raw session_id
            if episode_id:
                assert session_id not in episode_id or len(session_id) <= 8, \
                    f"Raw session_id leaked in episode_id for {description}"
                logger.info(f"✓ {description}: session_id properly sanitized")
            
        except Exception as e:
            logger.error(f"✗ Failed for {description}: {e}")
            return False
    
    logger.info("✓ Episodic memory session_id fix PASSED")
    return True


def test_tool_cache_serialization():
    """Test that tool cache handles non-serializable objects."""
    from agent.tool_cache import ToolCache
    
    logger.info("Testing tool cache JSON serialization fix...")
    
    cache = ToolCache()
    
    # Test with serializable args
    key1 = cache._generate_key("tool1", {"query": "test", "limit": 10})
    assert key1 is not None
    logger.info("✓ Serializable args handled correctly")
    
    # Test with non-serializable args (e.g., datetime object)
    non_serializable_args = {
        "query": "test",
        "timestamp": datetime.now(),  # Not JSON serializable
        "function": lambda x: x  # Not JSON serializable
    }
    
    try:
        key2 = cache._generate_key("tool2", non_serializable_args)
        assert key2 is not None
        logger.info("✓ Non-serializable args handled with fallback to repr")
    except Exception as e:
        logger.error(f"✗ Failed to handle non-serializable args: {e}")
        return False
    
    # Test cache operations with non-serializable args
    cache.set("tool3", non_serializable_args, "result")
    result = cache.get("tool3", non_serializable_args)
    assert result == "result"
    logger.info("✓ Cache operations work with non-serializable args")
    
    logger.info("✓ Tool cache serialization fix PASSED")
    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Running critical bug fix tests...")
    logger.info("=" * 60)
    
    results = []
    
    # Run all tests
    try:
        # Test 1: Optimized embedder metadata
        results.append(("Optimized Embedder Metadata", await test_optimized_embedder_metadata()))
        
        # Test 2: LRU cache behavior
        results.append(("LRU Cache Behavior", await test_lru_cache_behavior()))
        
        # Test 3: Agent __post_init__
        results.append(("Agent __post_init__", test_agent_post_init()))
        
        # Test 4: Episodic memory session_id
        results.append(("Episodic Memory Session ID", await test_episodic_memory_session_id()))
        
        # Test 5: Tool cache serialization
        results.append(("Tool Cache Serialization", test_tool_cache_serialization()))
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY:")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED - All critical bugs fixed!")
    else:
        logger.error("✗ SOME TESTS FAILED - Please review the fixes")
    logger.info("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)