"""
Embedding normalization module for dimension consistency.

This module is the central point for all embedding dimension normalization
across the system. It ensures all embeddings conform to the configured
target dimension, regardless of the source model or provider.

All components generating or processing embeddings MUST use this module
to ensure consistency.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

# Import centralized configuration
from agent.embedding_config import EmbeddingConfig

logger = logging.getLogger(__name__)


def truncate_embedding(embedding: List[float], target_dimension: Optional[int] = None) -> List[float]:
    """
    Truncate an embedding to a target dimension while preserving as much information as possible.
    
    This uses simple truncation which works well for most embeddings as they are often
    ordered by importance (earlier dimensions capture more variance).
    
    Args:
        embedding: Original embedding vector
        target_dimension: Target dimension size (uses configured default if not specified)
    
    Returns:
        Truncated embedding vector
    """
    if target_dimension is None:
        target_dimension = EmbeddingConfig.get_target_dimension()
    
    if len(embedding) <= target_dimension:
        return embedding
    
    # Simple truncation - take first N dimensions
    truncated = embedding[:target_dimension]
    
    # Optional: Renormalize to maintain unit length (improves similarity calculations)
    # This is recommended for cosine similarity searches
    norm = np.linalg.norm(truncated)
    if norm > 0:
        truncated = (np.array(truncated) / norm).tolist()
    
    logger.debug(f"Truncated embedding from {len(embedding)} to {target_dimension} dimensions")
    return truncated


def pad_embedding(embedding: List[float], target_dimension: Optional[int] = None) -> List[float]:
    """
    Pad an embedding to a target dimension if it's smaller.
    
    Args:
        embedding: Original embedding vector
        target_dimension: Target dimension size (uses configured default if not specified)
    
    Returns:
        Padded embedding vector
    """
    if target_dimension is None:
        target_dimension = EmbeddingConfig.get_target_dimension()
    
    if len(embedding) >= target_dimension:
        return embedding
    
    # Pad with zeros
    padded = embedding + [0.0] * (target_dimension - len(embedding))
    
    logger.debug(f"Padded embedding from {len(embedding)} to {target_dimension} dimensions")
    return padded


def normalize_embedding_dimension(embedding: List[float], target_dimension: Optional[int] = None, 
                                 model_name: Optional[str] = None) -> List[float]:
    """
    Normalize embedding to exact target dimension (truncate or pad as needed).
    
    This is the primary function that should be used by all components.
    
    Args:
        embedding: Original embedding vector
        target_dimension: Target dimension size (uses configured default if not specified)
        model_name: Optional model name for better logging
    
    Returns:
        Normalized embedding vector of exact target dimension
    """
    if target_dimension is None:
        target_dimension = EmbeddingConfig.get_target_dimension()
    
    current_dim = len(embedding)
    
    # Log dimension info if model name provided
    if model_name:
        EmbeddingConfig.log_dimension_info(model_name, current_dim)
    
    if current_dim == target_dimension:
        return embedding
    elif current_dim > target_dimension:
        normalized = truncate_embedding(embedding, target_dimension)
        logger.debug(f"Truncated embedding from {current_dim} to {target_dimension} dimensions")
        return normalized
    else:
        normalized = pad_embedding(embedding, target_dimension)
        logger.debug(f"Padded embedding from {current_dim} to {target_dimension} dimensions")
        return normalized


def validate_embedding_dimension(embedding: List[float], expected_dimension: Optional[int] = None,
                                raise_on_mismatch: bool = False) -> bool:
    """
    Validate that an embedding has the expected dimension.
    
    Args:
        embedding: Embedding vector to validate
        expected_dimension: Expected dimension (uses configured default if not specified)
        raise_on_mismatch: If True, raise ValueError on mismatch
    
    Returns:
        True if dimension matches, False otherwise
    
    Raises:
        ValueError: If raise_on_mismatch=True and dimensions don't match
    """
    if expected_dimension is None:
        expected_dimension = EmbeddingConfig.get_target_dimension()
    
    actual_dim = len(embedding)
    matches = actual_dim == expected_dimension
    
    if not matches:
        msg = f"Embedding dimension mismatch: expected {expected_dimension}, got {actual_dim}"
        if raise_on_mismatch:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    
    return matches


def batch_normalize_embeddings(embeddings: List[List[float]], 
                              target_dimension: Optional[int] = None,
                              model_name: Optional[str] = None) -> List[List[float]]:
    """
    Normalize a batch of embeddings to the target dimension.
    
    Args:
        embeddings: List of embedding vectors
        target_dimension: Target dimension (uses configured default if not specified)
        model_name: Optional model name for logging
    
    Returns:
        List of normalized embedding vectors
    """
    return [
        normalize_embedding_dimension(emb, target_dimension, model_name)
        for emb in embeddings
    ]


# Backward compatibility function
def get_target_dimension() -> int:
    """
    Get the target dimension from configuration.
    
    Deprecated: Use EmbeddingConfig.get_target_dimension() directly.
    """
    return EmbeddingConfig.get_target_dimension()