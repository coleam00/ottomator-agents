"""
Graph utilities for Neo4j/Graphiti integration.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import asyncio

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from dotenv import load_dotenv

# Import our clean patch module
from agent.graphiti_patch import apply_graphiti_embedding_patch
from agent.embedding_config import EmbeddingConfig

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Suppress Neo4j property warnings that are not actual errors
# These warnings occur when Graphiti checks for optional properties
neo4j_logger = logging.getLogger("neo4j")
neo4j_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Help from this PR for setting up the custom clients: https://github.com/getzep/graphiti/pull/601/files
class GraphitiClient:
    """Manages Graphiti knowledge graph operations."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        Initialize Graphiti client.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
        
        # LLM configuration - use instance variables for provider state
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        
        # Base URLs and models from environment
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_choice = os.getenv("LLM_CHOICE", "gpt-4.1-mini")
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        # Provider-aware embedding model defaults
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "gemini-embedding-001" if self.embedding_provider in ("gemini", "google") else "text-embedding-3-small"
        )
        
        # API keys - select based on provider
        self.llm_api_key = self._get_api_key_for_provider(self.llm_provider)
        self.embedding_api_key = self._get_api_key_for_provider(self.embedding_provider, is_embedding=True)
        
        if not self.llm_api_key:
            raise ValueError(f"API key not set for LLM provider: {self.llm_provider}")
        if not self.embedding_api_key:
            raise ValueError(f"API key not set for embedding provider: {self.embedding_provider}")
        
        # Import safe parsing function from models
        from .models import _safe_parse_int
        # Use 768 as default to meet Supabase limits and improve performance
        self.embedding_dimensions = _safe_parse_int("VECTOR_DIMENSION", 768, min_value=1, max_value=10000)
        
        self.graphiti: Optional[Graphiti] = None
        self.embedding_normalizer = None  # Track embedding normalizer
        self._initialized = False
    
    def _get_api_key_for_provider(self, provider: str, is_embedding: bool = False) -> Optional[str]:
        """Get the appropriate API key for a provider."""
        # For embeddings, check EMBEDDING_API_KEY first
        if is_embedding and os.getenv("EMBEDDING_API_KEY"):
            return os.getenv("EMBEDDING_API_KEY")
        
        # Check provider-specific keys
        if provider in ["gemini", "google"]:
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY")
        elif provider == "openai":
            return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        else:
            # For other providers (ollama, openrouter, etc.)
            return os.getenv("LLM_API_KEY")
    
    async def initialize(self):
        """Initialize Graphiti client."""
        if self._initialized:
            return
        
        try:
            # Use instance variables for provider configuration
            # These may have been modified by fallback logic
            llm_provider = self.llm_provider
            embedding_provider = self.embedding_provider
            
            # Configure LLM client based on provider
            if llm_provider in ["gemini", "google"]:
                # Use Gemini for LLM
                llm_config = LLMConfig(
                    api_key=self.llm_api_key,
                    model=self.llm_choice,
                    small_model=os.getenv("INGESTION_LLM_CHOICE", self.llm_choice)  # Use lighter model for small tasks
                )
                llm_client = GeminiClient(config=llm_config)
            else:
                # Default to OpenAI-compatible client
                llm_config = LLMConfig(
                    api_key=self.llm_api_key,
                    model=self.llm_choice,
                    small_model=self.llm_choice,
                    base_url=self.llm_base_url
                )
                llm_client = OpenAIClient(config=llm_config)
            
            # Get target dimension from centralized config
            target_dim = EmbeddingConfig.get_target_dimension()
            
            # Configure embedder based on provider
            if embedding_provider in ["gemini", "google"]:
                # Use Gemini for embeddings
                # IMPORTANT: Graphiti's embedding_dim parameter doesn't always work correctly
                # The patch applied below ensures proper normalization
                embedder = GeminiEmbedder(
                    config=GeminiEmbedderConfig(
                        api_key=self.embedding_api_key,
                        embedding_model=self.embedding_model,
                        embedding_dim=target_dim  # Use centralized dimension (768)
                    )
                )
            else:
                # Default to OpenAI embedder
                # IMPORTANT: Graphiti's embedding_dim parameter doesn't always work correctly
                # The patch applied below ensures proper normalization
                embedder = OpenAIEmbedder(
                    config=OpenAIEmbedderConfig(
                        api_key=self.embedding_api_key,
                        embedding_model=self.embedding_model,
                        embedding_dim=target_dim,  # Use centralized dimension (768)
                        base_url=self.embedding_base_url
                    )
                )
            
            # Configure cross-encoder/reranker based on LLM provider
            if llm_provider in ["gemini", "google"]:
                # Use Gemini reranker with the lighter model
                reranker_config = LLMConfig(
                    api_key=self.llm_api_key,
                    model=os.getenv("INGESTION_LLM_CHOICE", self.llm_choice)  # Use lighter model for reranking
                )
                cross_encoder = GeminiRerankerClient(config=reranker_config)
            else:
                # Default to OpenAI reranker
                cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)
            
            # Initialize Graphiti with configured clients
            self.graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder
            )
            
            # Build indices and constraints
            await self.graphiti.build_indices_and_constraints()
            
            # Apply embedding normalization patch if needed
            # This ensures all embeddings conform to the configured dimension
            # CRITICAL: This patch is essential for preventing dimension mismatches in Neo4j
            self.embedding_normalizer = apply_graphiti_embedding_patch(
                self.graphiti, 
                embedding_provider
            )
            
            if self.embedding_normalizer:
                logger.info(f"Embedding normalization active for {embedding_provider} - target dimension: {target_dim}")
            else:
                logger.warning(
                    f"Embedding normalization patch not applied for {embedding_provider}. "
                    f"This may cause dimension mismatches. Target: {target_dim}, "
                    f"Graphiti default: 1024"
                )
            
            self._initialized = True
            logger.info(f"Graphiti client initialized successfully with LLM: {self.llm_choice} and embedder: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise

    async def _fallback_to_openai_llm(self):
        """Reinitialize Graphiti forcing OpenAI LLM to avoid Gemini schema issues."""
        try:
            # Close existing graphiti if any
            if self.graphiti:
                await self.graphiti.close()
            
            # Update instance variables instead of mutating environment
            self.llm_provider = "openai"
            self.llm_api_key = self._get_api_key_for_provider("openai")
            
            if not self.llm_api_key:
                raise ValueError("OpenAI API key not available for fallback")
            
            self._initialized = False
            await self.initialize()
            logger.warning("Graphiti LLM provider fell back to OpenAI due to Gemini schema error")
        except Exception as e:
            logger.error(f"Failed to fallback to OpenAI LLM: {e}")
            raise
    
    async def close(self):
        """Close Graphiti connection."""
        if self.graphiti:
            # Remove embedding patch if applied
            if self.embedding_normalizer:
                self.embedding_normalizer.remove_patch()
                self.embedding_normalizer = None
            
            await self.graphiti.close()
            self.graphiti = None
            self._initialized = False
            logger.info("Graphiti client closed")
    
    async def add_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entity_types: Optional[Dict[str, Any]] = None,
        edge_types: Optional[Dict[str, Any]] = None,
        edge_type_map: Optional[Dict[tuple, List[str]]] = None,
        group_id: Optional[str] = None
    ):
        """
        Add an episode to the knowledge graph with optional custom entity types.
        
        Args:
            episode_id: Unique episode identifier
            content: Episode content
            source: Source of the content
            timestamp: Episode timestamp
            metadata: Additional metadata
            entity_types: Custom entity types for extraction
            edge_types: Custom edge types for relationships
            edge_type_map: Mapping of entity pairs to edge types
            group_id: Graph partition ID for user isolation (None for default, "0" for shared, user UUID for personal)
        """
        if not self._initialized:
            await self.initialize()
        
        episode_timestamp = timestamp or datetime.now(timezone.utc)
        
        # Import EpisodeType for proper source handling
        from graphiti_core.nodes import EpisodeType
        
        # Prepare kwargs for add_episode
        episode_kwargs = {
            "name": episode_id,
            "episode_body": content,
            "source": EpisodeType.text,  # Always use text type for our content
            "source_description": source,
            "reference_time": episode_timestamp
        }
        
        # Add group_id for graph partitioning if provided
        if group_id is not None:
            episode_kwargs["group_id"] = group_id
        
        # Add custom entity types if provided
        if entity_types:
            episode_kwargs["entity_types"] = entity_types
        
        if edge_types:
            episode_kwargs["edge_types"] = edge_types
            
        if edge_type_map:
            episode_kwargs["edge_type_map"] = edge_type_map
        
        # Note: Graphiti's add_episode does not accept a 'metadata' parameter directly
        # Append metadata as JSON to the episode content so it's included in the graph
        if metadata:
            import json
            metadata_str = f"\n\n[Metadata: {json.dumps(metadata, default=str)}]"
            episode_kwargs["episode_body"] = content + metadata_str
        
        try:
            await self.graphiti.add_episode(**episode_kwargs)
        except Exception as e:
            # Only fallback if we're currently using Gemini/Google provider
            if self.llm_provider in ["gemini", "google"]:
                # Handle Gemini response_schema additional_properties error by falling back
                err_str = str(e).lower()
                if "additional_properties" in err_str or "response_schema" in err_str or "invalid json payload" in err_str:
                    logger.error(f"Encountered Gemini schema error while adding episode, attempting fallback: {e}")
                    await self._fallback_to_openai_llm()
                    await self.graphiti.add_episode(**episode_kwargs)
                else:
                    raise
            else:
                # For non-Gemini providers, always re-raise the exception
                raise
        
        logger.info(f"Added episode {episode_id} to knowledge graph with custom entities: {bool(entity_types)}")
    
    async def add_episodes_bulk(
        self,
        bulk_episodes: List[Any],  # List[RawEpisode]
        group_id: Optional[str] = None,
        entity_types: Optional[Dict[str, Any]] = None,
        excluded_entity_types: Optional[List[str]] = None,
        edge_types: Optional[Dict[str, Any]] = None,
        edge_type_map: Optional[Dict[tuple, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Add multiple episodes to the knowledge graph in bulk.
        
        Args:
            bulk_episodes: List of RawEpisode objects to add
            group_id: Graph partition ID for user isolation
            entity_types: Custom entity types for extraction
            excluded_entity_types: Entity types to exclude from extraction
            edge_types: Custom edge types for relationships
            edge_type_map: Mapping of entity pairs to edge types
        
        Returns:
            Results dictionary with success/failure information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare kwargs for bulk ingestion
            kwargs = {"bulk_episodes": bulk_episodes}
            
            if group_id is not None:
                kwargs["group_id"] = group_id
            
            if entity_types:
                kwargs["entity_types"] = entity_types
            
            if excluded_entity_types:
                kwargs["excluded_entity_types"] = excluded_entity_types
            
            if edge_types:
                kwargs["edge_types"] = edge_types
            
            if edge_type_map:
                kwargs["edge_type_map"] = edge_type_map
            
            # Perform bulk ingestion
            await self.graphiti.add_episode_bulk(**kwargs)
            
            logger.info(f"Successfully added {len(bulk_episodes)} episodes in bulk")
            
            return {
                "success": True,
                "episodes_added": len(bulk_episodes),
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Failed to add episodes in bulk: {e}")
            return {
                "success": False,
                "episodes_added": 0,
                "errors": [str(e)]
            }
    
    async def search(
        self,
        query: str,
        center_node_distance: int = 2,
        use_hybrid_search: bool = True,
        group_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph.
        
        Args:
            query: Search query
            center_node_distance: Distance from center nodes
            use_hybrid_search: Whether to use hybrid search
            group_ids: List of group IDs to filter results (e.g., ["0"] for shared, [user_id] for personal)
        
        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build search kwargs
            search_kwargs = {"query": query}
            
            # Add group_ids filter if provided
            if group_ids is not None:
                search_kwargs["group_ids"] = group_ids
            
            # Use Graphiti's search method - embeddings are now normalized at initialization
            results = await self.graphiti.search(**search_kwargs)
            
            # Convert results to dictionaries
            return [
                {
                    "fact": result.fact,
                    "uuid": str(result.uuid),
                    "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                    "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None,
                    "source_node_uuid": str(result.source_node_uuid) if hasattr(result, 'source_node_uuid') and result.source_node_uuid else None
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def get_related_entities(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get entities related to a given entity using Graphiti search.
        
        Args:
            entity_name: Name of the entity
            relationship_types: Types of relationships to follow (not used with Graphiti)
            depth: Maximum depth to traverse (not used with Graphiti)
        
        Returns:
            Related entities and relationships
        """
        if not self._initialized:
            await self.initialize()
        
        # Use Graphiti search to find related information about the entity
        results = await self.graphiti.search(f"relationships involving {entity_name}")
        
        # Extract entity information from the search results
        related_entities = set()
        facts = []
        
        for result in results:
            facts.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None
            })
            
            # Simple entity extraction from fact text (could be enhanced)
            if entity_name.lower() in result.fact.lower():
                related_entities.add(entity_name)
        
        return {
            "central_entity": entity_name,
            "related_facts": facts,
            "search_method": "graphiti_semantic_search"
        }
    
    async def get_entity_timeline(
        self,
        entity_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of facts for an entity using Graphiti.
        
        Args:
            entity_name: Name of the entity
            start_date: Start of time range (not currently used)
            end_date: End of time range (not currently used)
        
        Returns:
            Timeline of facts
        """
        if not self._initialized:
            await self.initialize()
        
        # Search for temporal information about the entity
        results = await self.graphiti.search(f"timeline history of {entity_name}")
        
        timeline = []
        for result in results:
            timeline.append({
                "fact": result.fact,
                "uuid": str(result.uuid),
                "valid_at": str(result.valid_at) if hasattr(result, 'valid_at') and result.valid_at else None,
                "invalid_at": str(result.invalid_at) if hasattr(result, 'invalid_at') and result.invalid_at else None
            })
        
        # Sort by valid_at if available
        timeline.sort(key=lambda x: x.get('valid_at') or '', reverse=True)
        
        return timeline
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge graph.
        
        Returns:
            Graph statistics
        """
        if not self._initialized:
            await self.initialize()
        
        # For now, return a simple search to verify the graph is working
        # More detailed statistics would require direct Neo4j access
        try:
            test_results = await self.graphiti.search("test")
            return {
                "graphiti_initialized": True,
                "sample_search_results": len(test_results),
                "note": "Detailed statistics require direct Neo4j access"
            }
        except Exception as e:
            return {
                "graphiti_initialized": False,
                "error": str(e)
            }
    
    async def add_conversation_episode(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tools_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Add a conversation turn as an episode to the knowledge graph.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            tools_used: List of tools used
            metadata: Additional metadata
        
        Returns:
            Episode ID if successful
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate episode ID
            from uuid import uuid4
            episode_id = f"conversation_{session_id}_{uuid4().hex[:8]}"
            timestamp = datetime.now(timezone.utc)
            
            # Format conversation content
            content_parts = [
                f"User: {user_message}",
                f"Assistant: {assistant_response}"
            ]
            
            if tools_used:
                content_parts.append(f"Tools Used: {', '.join(tools_used)}")
            
            content = "\n\n".join(content_parts)
            
            # Add metadata
            episode_metadata = {
                "session_id": session_id,
                "conversation_turn": True,
                "tools_used": tools_used or [],
                **(metadata or {})
            }
            
            # Add to Graphiti
            await self.add_episode(
                episode_id=episode_id,
                content=content,
                source=f"conversation_session_{session_id}",
                timestamp=timestamp,
                metadata=episode_metadata
            )
            
            logger.info(f"Added conversation episode: {episode_id}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Failed to add conversation episode: {e}")
            return None
    
    async def get_session_episodes(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all episodes for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of episodes to return
        
        Returns:
            List of episodes from the session
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Search for episodes from this session
            results = await self.search(f"session {session_id} conversations")
            
            # Filter results to only those from this session
            session_episodes = []
            for result in results[:limit]:
                # Check if this result is from the session
                if session_id in result.get("fact", ""):
                    session_episodes.append(result)
            
            return session_episodes
            
        except Exception as e:
            logger.error(f"Failed to get session episodes: {e}")
            return []
    
    async def get_user_episodes(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all episodes for a specific user across sessions.
        
        Args:
            user_id: User identifier
            limit: Maximum number of episodes to return
        
        Returns:
            List of user's episodes
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Search for user-specific episodes
            results = await self.search(f"user {user_id} conversations history")
            
            # Return limited results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get user episodes: {e}")
            return []
    
    async def extract_conversation_facts(
        self,
        conversation: str,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract facts from a conversation using Graphiti.
        
        Args:
            conversation: Conversation text
            session_id: Optional session identifier
        
        Returns:
            List of extracted facts
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create a temporary episode to extract facts
            from uuid import uuid4
            temp_episode_id = f"fact_extraction_{uuid4().hex[:8]}"
            
            # Add episode (Graphiti will extract facts)
            await self.add_episode(
                episode_id=temp_episode_id,
                content=conversation,
                source=f"fact_extraction_{session_id}" if session_id else "fact_extraction",
                timestamp=datetime.now(timezone.utc)
            )
            
            # Search for the facts from this episode
            results = await self.search(conversation[:100])  # Use first 100 chars as query
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to extract conversation facts: {e}")
            return []
    
    async def add_fact_triples(
        self,
        triples: List[Tuple[str, str, str]],
        episode_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Add fact triples to the knowledge graph.
        
        Args:
            triples: List of (subject, predicate, object) tuples
            episode_id: Optional episode ID to associate with facts
            user_id: Optional user ID for user-specific facts (uses this as group_id)
        
        Returns:
            List of results for each triple (success/error status)
        """
        if not self._initialized:
            await self.initialize()
        
        results = []
        
        for subject, predicate, obj in triples:
            try:
                # Validate input
                if not all([subject, predicate, obj]):
                    results.append({
                        "triple": (subject, predicate, obj),
                        "status": "error",
                        "message": "Invalid triple: missing component"
                    })
                    continue
                
                # Import required Graphiti components
                from graphiti_core.nodes import EntityNode
                from graphiti_core.edges import EntityEdge
                import uuid
                
                # Use user_id for user-specific facts, or "0" for shared facts
                fact_group_id = user_id if user_id else "0"
                
                # Create nodes for subject and object
                subject_node = EntityNode(
                    uuid=str(uuid.uuid4()),
                    name=str(subject),
                    group_id=fact_group_id
                )
                
                object_node = EntityNode(
                    uuid=str(uuid.uuid4()),
                    name=str(obj),
                    group_id=fact_group_id
                )
                
                # Create edge for the relationship
                edge = EntityEdge(
                    group_id=fact_group_id,
                    source_node_uuid=subject_node.uuid,
                    target_node_uuid=object_node.uuid,
                    created_at=datetime.now(timezone.utc),
                    name=str(predicate),
                    fact=f"{subject} {predicate} {obj}"
                )
                
                # Add triplet to graph (Graphiti will handle deduplication)
                await self.graphiti.add_triplet(subject_node, edge, object_node)
                
                results.append({
                    "triple": (subject, predicate, obj),
                    "status": "success",
                    "message": "Added to graph"
                })
                
                logger.debug(f"Added fact triple: ({subject}, {predicate}, {obj})")
                
            except Exception as e:
                logger.error(f"Failed to add fact triple ({subject}, {predicate}, {obj}): {e}")
                results.append({
                    "triple": (subject, predicate, obj),
                    "status": "error",
                    "message": str(e)
                })
        
        return results
    
    async def clear_graph(self):
        """Clear all data from the graph (USE WITH CAUTION)."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Use Graphiti's proper clear_data function with the driver
            await clear_data(self.graphiti.driver)
            logger.warning("Cleared all data from knowledge graph")
        except Exception as e:
            logger.error(f"Failed to clear graph using clear_data: {e}")
            # Fallback: Close and reinitialize (this will create fresh indices)
            if self.graphiti:
                await self.graphiti.close()
            
            # Reinitialize with proper provider configuration
            self._initialized = False
            await self.initialize()
            
            logger.warning("Reinitialized Graphiti client (fresh indices created)")


# Global Graphiti client instance
graph_client = GraphitiClient()


async def initialize_graph():
    """Initialize graph client."""
    await graph_client.initialize()


async def close_graph():
    """Close graph client."""
    await graph_client.close()


# Convenience functions for common operations
async def add_to_knowledge_graph(
    content: str,
    source: str,
    episode_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    group_id: Optional[str] = None
) -> str:
    """
    Add content to the knowledge graph.
    
    Args:
        content: Content to add
        source: Source of the content
        episode_id: Optional episode ID
        metadata: Optional metadata
        group_id: Optional group ID for partitioning
    
    Returns:
        Episode ID
    """
    if not episode_id:
        episode_id = f"episode_{datetime.now(timezone.utc).isoformat()}"
    
    await graph_client.add_episode(
        episode_id=episode_id,
        content=content,
        source=source,
        metadata=metadata,
        group_id=group_id
    )
    
    return episode_id


async def search_knowledge_graph(
    query: str,
    group_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph.
    
    Args:
        query: Search query
        group_ids: Optional list of group IDs to filter by
    
    Returns:
        Search results
    """
    return await graph_client.search(query, group_ids=group_ids)


async def get_entity_relationships(
    entity: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get relationships for an entity.
    
    Args:
        entity: Entity name
        depth: Maximum traversal depth
    
    Returns:
        Entity relationships
    """
    return await graph_client.get_related_entities(entity, depth=depth)


async def test_graph_connection() -> bool:
    """
    Test graph database connection.
    
    Returns:
        True if connection successful
    """
    try:
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Graph connection test failed: {e}")
        return False