"""
Embedding truncator for compatibility with database dimension limits.

This module handles normalization of embeddings to a standardized dimension.
All embeddings are normalized to 768 dimensions,
regardless of the source model's native dimensions (e.g., Gemini's 3072, OpenAI's 1536).
"""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


def truncate_embedding(embedding: List[float], target_dimension: int = 768) -> List[float]:
    """
    Truncate an embedding to a target dimension while preserving as much information as possible.
    
    This uses simple truncation which works well for most embeddings as they are often
    ordered by importance (earlier dimensions capture more variance).
    
    Args:
        embedding: Original embedding vector
        target_dimension: Target dimension size (default 768 for standardization)
    
    Returns:
        Truncated embedding vector
    """
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


def pad_embedding(embedding: List[float], target_dimension: int = 768) -> List[float]:
    """
    Pad an embedding to a target dimension if it's smaller.
    
    Args:
        embedding: Original embedding vector
        target_dimension: Target dimension size
    
    Returns:
        Padded embedding vector
    """
    if len(embedding) >= target_dimension:
        return embedding
    
    # Pad with zeros
    padded = embedding + [0.0] * (target_dimension - len(embedding))
    
    logger.debug(f"Padded embedding from {len(embedding)} to {target_dimension} dimensions")
    return padded


def normalize_embedding_dimension(embedding: List[float], target_dimension: int = 768) -> List[float]:
    """
    Normalize embedding to exact target dimension (truncate or pad as needed).
    
    Args:
        embedding: Original embedding vector
        target_dimension: Target dimension size
    
    Returns:
        Normalized embedding vector of exact target dimension
    """
    current_dim = len(embedding)
    
    if current_dim == target_dimension:
        return embedding
    elif current_dim > target_dimension:
        return truncate_embedding(embedding, target_dimension)
    else:
        return pad_embedding(embedding, target_dimension)


# Configuration based on environment
def get_target_dimension() -> int:
    """Get the target dimension from environment or default."""
    import os
    from agent.models import _safe_parse_int
    
    # Default to 768 to match database schema (Gemini with truncation)
    return _safe_parse_int("VECTOR_DIMENSION", 768, min_value=1, max_value=10000)