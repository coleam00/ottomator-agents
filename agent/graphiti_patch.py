"""
Graphiti embedding normalization patch.

This module provides a clean patch for Graphiti's embedding generation
to ensure all embeddings conform to the configured target dimension.

The patch is necessary because Graphiti's GeminiEmbedderConfig's 
embedding_dim parameter is not always respected properly.
"""

import logging
from typing import Any, Optional, Callable
from ingestion.embedding_truncator import normalize_embedding_dimension
from agent.embedding_config import EmbeddingConfig

logger = logging.getLogger(__name__)


class GraphitiEmbeddingNormalizer:
    """
    Wrapper class for Graphiti embedder to ensure dimension normalization.
    """
    
    def __init__(self, embedder: Any, provider: str = "unknown"):
        """
        Initialize the normalizer wrapper.
        
        Args:
            embedder: The original Graphiti embedder instance
            provider: Name of the embedding provider for logging
        """
        self.embedder = embedder
        self.provider = provider
        self.target_dimension = EmbeddingConfig.get_target_dimension()
        self._original_create = None
        self._patched = False
    
    def apply_patch(self):
        """
        Apply the normalization patch to the embedder's create method.
        """
        if self._patched:
            logger.debug("Patch already applied to embedder")
            return
        
        if not hasattr(self.embedder, 'create'):
            logger.warning(f"Embedder {self.provider} does not have 'create' method, skipping patch")
            return
        
        # Store original method
        self._original_create = self.embedder.create
        
        # Create normalized wrapper
        async def normalized_create(text=None, input_data=None, **kwargs):
            """
            Wrapper that ensures embeddings are normalized to target dimension.
            
            Handles both calling styles used by Graphiti:
            - Some parts call with text parameter
            - Others call with input_data parameter
            """
            # Determine actual text input
            actual_text = text if text is not None else input_data
            if actual_text is None:
                raise ValueError("Either 'text' or 'input_data' must be provided")
            
            # Generate embedding using original method
            embedding = await self._original_create(actual_text, **kwargs)
            
            # Get text preview for logging
            text_preview = str(actual_text)[:50] + "..." if len(str(actual_text)) > 50 else str(actual_text)
            
            # Normalize embedding
            original_dim = len(embedding)
            normalized = normalize_embedding_dimension(
                embedding, 
                self.target_dimension,
                model_name=self.provider
            )
            
            if original_dim != self.target_dimension:
                logger.debug(
                    f"{self.provider}: Normalized embedding from {original_dim} to "
                    f"{self.target_dimension} dimensions for text: '{text_preview}'"
                )
            
            return normalized
        
        # Replace method with normalized version
        self.embedder.create = normalized_create
        self._patched = True
        
        logger.info(
            f"Applied dimension normalization patch to {self.provider} embedder "
            f"(target: {self.target_dimension} dimensions)"
        )
    
    def remove_patch(self):
        """
        Remove the normalization patch and restore original method.
        """
        if not self._patched or not self._original_create:
            logger.debug("No patch to remove")
            return
        
        self.embedder.create = self._original_create
        self._patched = False
        logger.info(f"Removed dimension normalization patch from {self.provider} embedder")


def apply_graphiti_embedding_patch(graphiti_client: Any, provider: str) -> Optional[GraphitiEmbeddingNormalizer]:
    """
    Apply embedding normalization patch to a Graphiti client.
    
    This function checks if the patch is needed based on the provider
    and applies it if necessary.
    
    Args:
        graphiti_client: The Graphiti client instance
        provider: The embedding provider name
    
    Returns:
        GraphitiEmbeddingNormalizer instance if patch was applied, None otherwise
    """
    # Only patch providers known to have dimension issues
    providers_needing_patch = ["gemini", "google", "openai"]
    
    # Clean provider name for matching
    provider_lower = provider.lower().strip()
    
    needs_patch = any(p in provider_lower for p in providers_needing_patch)
    
    if not needs_patch:
        logger.debug(f"Provider {provider} does not require embedding patch")
        return None
    
    if not hasattr(graphiti_client, 'embedder'):
        logger.warning("Graphiti client does not have embedder attribute, cannot apply patch")
        return None
    
    try:
        normalizer = GraphitiEmbeddingNormalizer(graphiti_client.embedder, provider)
        normalizer.apply_patch()
        return normalizer
    except Exception as e:
        logger.error(f"Failed to apply embedding patch: {e}")
        return None