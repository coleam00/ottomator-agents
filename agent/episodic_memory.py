"""
Episodic memory service for managing conversation memories in Graphiti.
"""

import os
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

from .graph_utils import graph_client
from .models import Message
from .medical_entities import (
    medical_entity_extractor,
    get_medical_entity_types,
    get_medical_edge_types,
    get_medical_edge_type_map
)

logger = logging.getLogger(__name__)


class EpisodicMemoryService:
    """Service for managing episodic memories in Graphiti."""
    
    def __init__(self):
        self.graph_client = graph_client
        self._enabled = os.getenv("ENABLE_EPISODIC_MEMORY", "true").lower() == "true"
        self._batch_threshold = int(os.getenv("EPISODIC_BATCH_SIZE", "5"))
        self._pending_episodes = []
        self._fallback_dir = Path(os.getenv("EPISODIC_FALLBACK_PATH", "./failed_episodes"))
        self._fallback_dir.mkdir(exist_ok=True)
        self._extract_medical_entities = os.getenv("MEDICAL_ENTITY_EXTRACTION", "true").lower() == "true"
        self._fact_confidence_threshold = float(os.getenv("FACT_EXTRACTION_CONFIDENCE", "0.7"))
        self.entity_extractor = medical_entity_extractor
        
    async def create_conversation_episode(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tools_used: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create an episodic memory from a conversation turn.
        
        This method creates a rich episodic memory that includes:
        - User query and context
        - Assistant's response and reasoning
        - Tools used and their results
        - Temporal information
        - Medical entities and relationships
        - Fact triples for knowledge graph
        """
        if not self._enabled:
            logger.debug("Episodic memory disabled")
            return None
            
        episode_id = f"conversation_{session_id}_{uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc)
        
        # Extract medical entities and facts
        entities = {}
        facts = []
        
        if self._extract_medical_entities:
            # Extract entities from both messages
            user_entities = self.extract_medical_entities(user_message)
            assistant_entities = self.extract_medical_entities(assistant_response)
            
            # Combine entities
            entities = {
                "user_entities": user_entities,
                "assistant_entities": assistant_entities
            }
            
            # Extract fact triples
            facts = self.extract_fact_triples(user_message, assistant_response)
            
            # Track symptom timeline if symptoms are mentioned
            for symptom in user_entities.get("symptoms", []):
                await self.create_symptom_timeline(
                    session_id=session_id,
                    symptom=symptom["name"],
                    timestamp=timestamp,
                    severity=symptom.get("severity"),
                    metadata={"episode_id": episode_id}
                )
        
        # Calculate importance score
        importance_score = self.calculate_memory_importance(
            entities=entities,
            facts=facts,
            metadata=metadata
        )
        
        # Construct rich episode content
        episode_content = self._format_episode_content(
            user_message=user_message,
            assistant_response=assistant_response,
            tools_used=tools_used,
            metadata=metadata
        )
        
        # Source description for the episode
        source_description = f"User conversation in session {session_id}"
        
        # Enhanced metadata with medical information
        enhanced_metadata = {
            "session_id": session_id,
            "user_id": metadata.get("user_id") if metadata else None,
            "conversation_turn": True,
            "tools_used": [tool.get("tool_name") for tool in (tools_used or [])],
            "medical_entities": entities,
            "fact_triples": facts,
            "importance_score": importance_score,
            **(metadata or {})
        }
        
        try:
            # Get medical entity types for Graphiti if enabled
            entity_types = None
            edge_types = None
            edge_type_map = None
            
            if self._extract_medical_entities:
                entity_types = get_medical_entity_types()
                edge_types = get_medical_edge_types()
                edge_type_map = get_medical_edge_type_map()
            
            # Add episode with custom entity types
            await self.graph_client.add_episode(
                episode_id=episode_id,
                content=episode_content,
                source=source_description,
                timestamp=timestamp,
                metadata=enhanced_metadata,
                entity_types=entity_types,
                edge_types=edge_types,
                edge_type_map=edge_type_map
            )
            
            # Store fact triples in knowledge graph
            if facts:
                try:
                    results = await self.graph_client.add_fact_triples(facts, episode_id)
                    success_count = sum(1 for r in results if r["status"] == "success")
                    logger.info(f"Added {success_count}/{len(facts)} fact triples to knowledge graph")
                    
                    # Log any failures
                    for result in results:
                        if result["status"] == "error":
                            logger.warning(f"Failed to add fact triple {result['triple']}: {result['message']}")
                except Exception as e:
                    logger.error(f"Error adding fact triples to graph: {e}")
                    # Continue - don't fail the episode creation due to fact storage issues
            
            logger.info(f"Created episodic memory: {episode_id} with importance: {importance_score}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Failed to create episodic memory: {e}")
            # Save to fallback storage
            await self._save_failed_episode(
                episode_id=episode_id,
                content=episode_content,
                source=source_description,
                timestamp=timestamp,
                metadata=enhanced_metadata,
                error=str(e)
            )
            return None
    
    def _format_episode_content(
        self,
        user_message: str,
        assistant_response: str,
        tools_used: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format conversation into structured episode content."""
        
        # Build structured content for better fact extraction
        content_parts = [
            f"User Query: {user_message}",
            f"Assistant Response: {assistant_response}"
        ]
        
        if tools_used:
            tool_summary = ", ".join([t.get("tool_name", "unknown") for t in tools_used])
            content_parts.append(f"Tools Used: {tool_summary}")
        
        if metadata and metadata.get("search_type"):
            content_parts.append(f"Search Type: {metadata['search_type']}")
        
        return "\n\n".join(content_parts)
    
    def extract_medical_entities(self, text: str) -> Dict[str, Any]:
        """Extract medical entities from text."""
        if not self._extract_medical_entities:
            return {}
        
        try:
            entities = self.entity_extractor.extract_all_entities(text)
            logger.debug(f"Extracted medical entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Failed to extract medical entities: {e}")
            return {}
    
    def extract_fact_triples(
        self, 
        user_message: str, 
        assistant_response: str
    ) -> List[Tuple[str, str, str]]:
        """Extract fact triples from conversation."""
        facts = []
        
        # Extract medical entities from both messages
        user_entities = self.extract_medical_entities(user_message)
        assistant_entities = self.extract_medical_entities(assistant_response)
        
        # Create fact triples from extracted entities
        # Example: (Patient, HAS_SYMPTOM, "headache")
        for symptom in user_entities.get("symptoms", []):
            facts.append(("Patient", "HAS_SYMPTOM", symptom["name"]))
            if symptom.get("location"):
                facts.append((symptom["name"], "LOCATED_IN", symptom["location"]))
            if symptom.get("severity"):
                facts.append((symptom["name"], "HAS_SEVERITY", symptom["severity"]))
        
        for condition in user_entities.get("conditions", []):
            facts.append(("Patient", "MAY_HAVE_CONDITION", condition["name"]))
        
        for treatment in assistant_entities.get("treatments", []):
            facts.append(("Assistant", "RECOMMENDS_TREATMENT", treatment["name"]))
        
        # Extract temporal facts with case-insensitive patterns
        temporal_patterns = [
            (r"started (\d+ \w+ ago)", "SYMPTOM_ONSET"),
            (r"for (\d+ \w+)", "SYMPTOM_DURATION"),
            (r"since (\w+)", "SYMPTOM_SINCE")
        ]
        
        import re
        for pattern, predicate in temporal_patterns:
            # Use IGNORECASE flag for proper case-insensitive matching
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match and match.group(1):
                facts.append(("Symptom", predicate, match.group(1)))
        
        return facts
    
    async def create_symptom_timeline(
        self,
        session_id: str,
        symptom: str,
        timestamp: datetime,
        severity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Track symptom progression over time."""
        timeline_id = f"symptom_timeline_{session_id}_{uuid4().hex[:8]}"
        
        content = f"Symptom Timeline Entry: {symptom}"
        if severity:
            content += f" - Severity: {severity}"
        
        try:
            await self.graph_client.add_episode(
                episode_id=timeline_id,
                content=content,
                source=f"symptom_timeline_{session_id}",
                timestamp=timestamp,
                metadata={
                    "session_id": session_id,
                    "symptom": symptom,
                    "severity": severity,
                    "timeline_entry": True,
                    **(metadata or {})
                }
            )
            
            logger.info(f"Created symptom timeline entry: {timeline_id}")
            return timeline_id
            
        except Exception as e:
            logger.error(f"Failed to create symptom timeline: {e}")
            return None
    
    def calculate_memory_importance(
        self,
        entities: Dict[str, Any],
        facts: List[Tuple[str, str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate importance score for episodic memory."""
        score = 0.5  # Base score
        
        # Increase score based on medical entities
        entity_weights = {
            "symptoms": 0.15,
            "conditions": 0.2,
            "treatments": 0.15,
            "body_parts": 0.05
        }
        
        for entity_type, weight in entity_weights.items():
            if entities.get(entity_type):
                score += weight * min(len(entities[entity_type]) / 3, 1.0)
        
        # Increase score based on fact count
        if facts:
            score += 0.2 * min(len(facts) / 5, 1.0)
        
        # Increase score for critical symptoms
        critical_keywords = ["severe", "emergency", "urgent", "critical", "unbearable"]
        if metadata and any(keyword in str(metadata).lower() for keyword in critical_keywords):
            score += 0.25
        
        return min(score, 1.0)
    
    async def create_batch_episode(
        self,
        session_id: str,
        messages: List[Message]
    ) -> Optional[str]:
        """Create episodic memory from multiple conversation turns."""
        if not messages or not self._enabled:
            return None
            
        episode_id = f"batch_conversation_{session_id}_{uuid4().hex[:8]}"
        
        # Format all messages into a cohesive episode
        content_parts = []
        for msg in messages:
            content_parts.append(f"{msg.role.upper()}: {msg.content}")
        
        episode_content = "\n\n".join(content_parts)
        
        try:
            await self.graph_client.add_episode(
                episode_id=episode_id,
                content=episode_content,
                source=f"Batch conversation from session {session_id}",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "session_id": session_id,
                    "message_count": len(messages),
                    "batch_episode": True
                }
            )
            
            logger.info(f"Created batch episode: {episode_id}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Failed to create batch episode: {e}")
            return None
    
    async def search_episodic_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memories with optional filtering.
        """
        if not self._enabled:
            return []
            
        # Build search query with context
        search_query = query
        if session_id:
            search_query = f"{query} session:{session_id}"
        elif user_id:
            search_query = f"{query} user:{user_id}"
        
        try:
            results = await self.graph_client.search(search_query)
            
            # Filter to conversation episodes
            conversation_results = [
                r for r in results 
                if isinstance(r, dict) and "conversation" in str(r.get("source_node_uuid", "")).lower()
            ]
            
            return conversation_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search episodic memories: {e}")
            return []
    
    async def get_session_timeline(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Get temporal timeline of a session's episodic memories."""
        if not self._enabled:
            return []
            
        try:
            # Search for all episodes from this session
            results = await self.graph_client.search(f"session {session_id}")
            
            # Sort by temporal validity
            timeline = sorted(
                results,
                key=lambda x: x.get("valid_at", ""),
                reverse=True
            )
            
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to get session timeline: {e}")
            return []
    
    async def get_user_memories(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all episodic memories for a specific user across sessions."""
        if not self._enabled:
            return []
            
        try:
            # Search for user-specific memories
            results = await self.graph_client.search(f"user {user_id} conversations")
            
            # Sort by recency
            user_memories = sorted(
                results,
                key=lambda x: x.get("valid_at", ""),
                reverse=True
            )
            
            return user_memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get user memories: {e}")
            return []
    
    async def _save_failed_episode(
        self,
        episode_id: str,
        content: str,
        source: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]],
        error: str
    ):
        """Save failed episode to disk for later retry."""
        filename = f"{episode_id}.json"
        filepath = self._fallback_dir / filename
        
        data = {
            "episode_id": episode_id,
            "content": content,
            "source": source,
            "timestamp": timestamp.isoformat(),
            "metadata": metadata,
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved failed episode to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save episode to fallback storage: {e}")
    
    async def retry_failed_episodes(self) -> int:
        """Retry creating failed episodes from fallback storage."""
        if not self._enabled:
            return 0
            
        retry_count = 0
        
        for filepath in self._fallback_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Retry creation
                await self.graph_client.add_episode(
                    episode_id=data['episode_id'],
                    content=data['content'],
                    source=data['source'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    metadata=data.get('metadata', {})
                )
                
                # Remove file on success
                filepath.unlink()
                logger.info(f"Successfully retried episode from {filepath}")
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Failed to retry episode from {filepath}: {e}")
        
        return retry_count


class EpisodicMemoryQueue:
    """Queue for batch processing episodic memories."""
    
    def __init__(self, batch_size: int = 5, flush_interval: float = 30.0):
        self.queue = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._lock = asyncio.Lock()
        self._flush_task = None
        
    async def add(self, episode_data: Dict[str, Any]):
        """Add episode to queue."""
        async with self._lock:
            self.queue.append(episode_data)
            
            if len(self.queue) >= self.batch_size:
                await self.flush()
    
    async def flush(self):
        """Process all queued episodes."""
        async with self._lock:
            if not self.queue:
                return
                
            episodes_to_process = self.queue.copy()
            self.queue.clear()
        
        # Process episodes with proper error handling and retry
        failed_episodes = []
        tasks = [
            self._create_episode_with_retry(episode)
            for episode in episodes_to_process
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to create episode after retries: {result}")
                failed_episodes.append(episodes_to_process[i])
        
        # Store failed episodes for later retry
        if failed_episodes:
            await self._store_failed_episodes(failed_episodes)
    
    async def _create_episode(self, episode_data: Dict[str, Any]):
        """Create a single episode."""
        return await graph_client.add_episode(**episode_data)
    
    async def _create_episode_with_retry(self, episode_data: Dict[str, Any], max_retries: int = 3):
        """Create a single episode with retry logic."""
        for attempt in range(max_retries):
            try:
                return await self._create_episode(episode_data)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Episode creation attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise  # Re-raise on final attempt
    
    async def _store_failed_episodes(self, episodes: List[Dict[str, Any]]):
        """Store failed episodes for later retry."""
        import json
        from pathlib import Path
        from datetime import datetime, timezone
        
        fallback_dir = Path("./failed_episodes")
        fallback_dir.mkdir(exist_ok=True)
        
        for episode in episodes:
            try:
                filename = f"failed_episode_{datetime.now(timezone.utc).isoformat()}_{episode.get('episode_id', 'unknown')}.json"
                filepath = fallback_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(episode, f, indent=2, default=str)
                
                logger.info(f"Stored failed episode to {filepath}")
            except Exception as e:
                logger.error(f"Failed to store episode to fallback: {e}")
    
    async def start_auto_flush(self):
        """Start automatic flush timer."""
        self._flush_task = asyncio.create_task(self._auto_flush_loop())
    
    async def _auto_flush_loop(self):
        """Periodically flush the queue."""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush()
    
    async def stop_auto_flush(self):
        """Stop the auto-flush task."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass


# Global service instance
episodic_memory_service = EpisodicMemoryService()

# Global queue instance
episodic_queue = EpisodicMemoryQueue(
    batch_size=int(os.getenv("EPISODIC_BATCH_SIZE", "5")),
    flush_interval=float(os.getenv("EPISODIC_FLUSH_INTERVAL", "30"))
)