"""
Centralized configuration for embedding dimensions.

This module provides a single source of truth for embedding dimension configuration
across the entire system. All components should use this module to ensure consistency.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingConfig:
    """Central configuration for embedding dimensions and normalization."""
    
    # Standard dimension for all embeddings in the system
    # Set to 768 for optimal balance between performance and quality
    STANDARD_DIMENSION = 768
    
    # Known embedding model dimensions
    MODEL_DIMENSIONS = {
        # OpenAI models
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        
        # Google/Gemini models
        "gemini-embedding-001": 3072,
        "embedding-001": 3072,  # Alias
        "models/text-embedding-004": 768,
        
        # Ollama models
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        
        # Cohere models
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
    }
    
    @classmethod
    def get_target_dimension(cls) -> int:
        """
        Get the target dimension for embeddings.
        
        Can be overridden by EMBEDDING_DIMENSION environment variable,
        but defaults to STANDARD_DIMENSION (768) for consistency.
        
        Returns:
            Target dimension for embeddings
        """
        env_dim = os.getenv("EMBEDDING_DIMENSION")
        if env_dim:
            try:
                custom_dim = int(env_dim)
                if custom_dim != cls.STANDARD_DIMENSION:
                    logger.warning(
                        f"Using custom embedding dimension {custom_dim} instead of standard {cls.STANDARD_DIMENSION}. "
                        "Ensure your database schema matches this dimension."
                    )
                return custom_dim
            except ValueError:
                logger.error(f"Invalid EMBEDDING_DIMENSION value: {env_dim}. Using standard dimension.")
        
        return cls.STANDARD_DIMENSION
    
    @classmethod
    def get_model_native_dimension(cls, model_name: str) -> Optional[int]:
        """
        Get the native dimension for a specific embedding model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Native dimension of the model, or None if unknown
        """
        # Clean up model name for matching
        model_key = model_name.lower().strip()
        
        # Check for exact match
        if model_key in cls.MODEL_DIMENSIONS:
            return cls.MODEL_DIMENSIONS[model_key]
        
        # Check for partial matches
        for key, dim in cls.MODEL_DIMENSIONS.items():
            if key in model_key or model_key in key:
                return dim
        
        return None
    
    @classmethod
    def log_dimension_info(cls, model_name: str, actual_dim: int):
        """
        Log information about embedding dimensions for debugging.
        
        Args:
            model_name: Name of the embedding model
            actual_dim: Actual dimension of generated embedding
        """
        target_dim = cls.get_target_dimension()
        expected_dim = cls.get_model_native_dimension(model_name)
        
        if expected_dim and expected_dim != actual_dim:
            logger.warning(
                f"Model '{model_name}' generated {actual_dim}-dim embedding, "
                f"expected {expected_dim}-dim based on model specs"
            )
        
        if actual_dim != target_dim:
            action = "truncated" if actual_dim > target_dim else "padded"
            logger.info(
                f"Embedding will be {action} from {actual_dim} to {target_dim} dimensions"
            )
        elif actual_dim == target_dim:
            logger.debug(f"Embedding already at target dimension: {target_dim}")


# Convenience function for backward compatibility
def get_embedding_dimension() -> int:
    """Get the target embedding dimension."""
    return EmbeddingConfig.get_target_dimension()