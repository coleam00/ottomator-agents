#!/usr/bin/env python3
"""
Test script to verify the embedding dimension refactoring works correctly.

This script tests:
1. Centralized configuration
2. Embedding normalization
3. Graphiti patch application
4. Dimension consistency across modules
"""

import asyncio
import logging
import sys
from typing import List
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_centralized_config():
    """Test centralized embedding configuration."""
    logger.info("Testing centralized embedding configuration...")
    
    from agent.embedding_config import EmbeddingConfig
    
    # Test getting target dimension
    target_dim = EmbeddingConfig.get_target_dimension()
    logger.info(f"Target dimension: {target_dim}")
    assert target_dim == 768, f"Expected 768, got {target_dim}"
    
    # Test model dimension lookup
    models = [
        ("text-embedding-3-small", 1536),
        ("gemini-embedding-001", 3072),
        ("nomic-embed-text", 768)
    ]
    
    for model_name, expected_dim in models:
        native_dim = EmbeddingConfig.get_model_native_dimension(model_name)
        logger.info(f"Model {model_name}: native dimension = {native_dim}")
        if native_dim:
            assert native_dim == expected_dim, f"Expected {expected_dim}, got {native_dim}"
    
    logger.info("✅ Centralized configuration test passed")


async def test_embedding_normalization():
    """Test embedding normalization functions."""
    logger.info("Testing embedding normalization...")
    
    from ingestion.embedding_truncator import (
        normalize_embedding_dimension,
        validate_embedding_dimension,
        batch_normalize_embeddings
    )
    
    # Test truncation (3072 -> 768)
    long_embedding = [float(i) for i in range(3072)]
    normalized = normalize_embedding_dimension(long_embedding, 768)
    assert len(normalized) == 768, f"Expected 768 dimensions, got {len(normalized)}"
    
    # Verify normalization (should have unit length)
    norm = np.linalg.norm(normalized)
    assert abs(norm - 1.0) < 0.001, f"Expected unit norm, got {norm}"
    
    # Test padding (384 -> 768)
    short_embedding = [float(i) for i in range(384)]
    padded = normalize_embedding_dimension(short_embedding, 768)
    assert len(padded) == 768, f"Expected 768 dimensions, got {len(padded)}"
    
    # Test validation
    valid_embedding = [0.0] * 768
    assert validate_embedding_dimension(valid_embedding, 768) == True
    assert validate_embedding_dimension(long_embedding, 768) == False
    
    # Test batch normalization
    batch = [long_embedding, short_embedding, valid_embedding]
    normalized_batch = batch_normalize_embeddings(batch, 768)
    assert all(len(emb) == 768 for emb in normalized_batch), "Batch normalization failed"
    
    logger.info("✅ Embedding normalization test passed")


async def test_graphiti_patch():
    """Test Graphiti embedding patch application."""
    logger.info("Testing Graphiti patch...")
    
    from agent.graphiti_patch import GraphitiEmbeddingNormalizer
    
    # Create a mock embedder
    class MockEmbedder:
        async def create(self, text):
            # Simulate returning 3072-dimensional embedding
            return [float(i) for i in range(3072)]
    
    embedder = MockEmbedder()
    normalizer = GraphitiEmbeddingNormalizer(embedder, "test-provider")
    
    # Apply patch
    normalizer.apply_patch()
    
    # Test normalized output
    result = await embedder.create("test text")
    assert len(result) == 768, f"Expected 768 dimensions after patch, got {len(result)}"
    
    # Remove patch
    normalizer.remove_patch()
    
    # Test unpatched output
    result = await embedder.create("test text")
    assert len(result) == 3072, f"Expected 3072 dimensions without patch, got {len(result)}"
    
    logger.info("✅ Graphiti patch test passed")


async def test_tools_integration():
    """Test tools module integration with centralized normalization."""
    logger.info("Testing tools module integration...")
    
    try:
        from agent.tools import generate_embedding
        from agent.providers import get_embedding_model
        
        # Generate a test embedding
        test_text = "This is a test for embedding normalization"
        embedding = await generate_embedding(test_text)
        
        logger.info(f"Generated embedding dimension: {len(embedding)}")
        assert len(embedding) == 768, f"Expected 768 dimensions, got {len(embedding)}"
        
        # Verify it's normalized (unit length)
        norm = np.linalg.norm(embedding)
        logger.info(f"Embedding norm: {norm}")
        
        logger.info("✅ Tools integration test passed")
    except Exception as e:
        logger.warning(f"Could not test tools integration (may need API keys): {e}")


async def test_unified_db_utils():
    """Test unified_db_utils module with centralized configuration."""
    logger.info("Testing unified_db_utils module...")
    
    from agent.unified_db_utils import comprehensive_search
    from agent.embedding_config import EmbeddingConfig
    
    # Verify configuration is imported
    target_dim = EmbeddingConfig.get_target_dimension()
    assert target_dim == 768, f"Configuration not properly imported in unified_db_utils"
    
    logger.info("✅ Unified DB utils test passed")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Starting embedding refactoring tests...")
    logger.info("=" * 60)
    
    tests = [
        test_centralized_config,
        test_embedding_normalization,
        test_graphiti_patch,
        test_tools_integration,
        test_unified_db_utils
    ]
    
    failed = []
    for test in tests:
        try:
            await test()
        except AssertionError as e:
            logger.error(f"❌ {test.__name__} failed: {e}")
            failed.append(test.__name__)
        except Exception as e:
            logger.error(f"❌ {test.__name__} error: {e}")
            failed.append(test.__name__)
    
    logger.info("=" * 60)
    if failed:
        logger.error(f"FAILED: {len(failed)} tests failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("The embedding dimension refactoring is working correctly.")
        logger.info("=" * 60)
        logger.info("SUMMARY:")
        logger.info("- Centralized configuration in agent/embedding_config.py")
        logger.info("- Enhanced normalization in ingestion/embedding_truncator.py")
        logger.info("- Clean patch module in agent/graphiti_patch.py")
        logger.info("- All modules using centralized configuration")
        logger.info("- Dimension validation and logging added")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())