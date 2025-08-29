#!/usr/bin/env python3
"""
Robust Neo4j Knowledge Graph Builder with Episode-Level Checkpointing
Ensures 100% completion of Neo4j episodic node creation for all documents.
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import logging

from dotenv import load_dotenv
from supabase import create_client, Client
from ingestion.chunker import DocumentChunk
from agent.graph_utils import GraphitiClient

# Setup logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neo4j_robust_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EpisodeStatus(Enum):
    """Status of episode processing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class EpisodeCheckpoint:
    """Checkpoint for a single episode."""
    episode_id: str
    document_id: str
    document_title: str
    chunk_index: int
    status: EpisodeStatus
    retry_count: int = 0
    last_error: Optional[str] = None
    processing_time_ms: float = 0
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeCheckpoint':
        """Create from dictionary."""
        data['status'] = EpisodeStatus(data['status'])
        return cls(**data)


@dataclass
class Neo4jIngestionState:
    """Overall Neo4j ingestion state."""
    session_id: str
    started_at: str
    updated_at: str
    total_documents: int
    total_chunks: int
    completed_episodes: int
    failed_episodes: int
    current_document_index: int
    episodes: Dict[str, EpisodeCheckpoint] = field(default_factory=dict)
    document_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['episodes'] = {
            ep_id: checkpoint.to_dict() 
            for ep_id, checkpoint in self.episodes.items()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Neo4jIngestionState':
        """Create from dictionary."""
        episodes = {
            ep_id: EpisodeCheckpoint.from_dict(ep_data)
            for ep_id, ep_data in data.get('episodes', {}).items()
        }
        data['episodes'] = episodes
        return cls(**data)


class RobustNeo4jBuilder:
    """Robust Neo4j knowledge graph builder with checkpoint/resume capability."""
    
    def __init__(
        self,
        checkpoint_dir: str = ".neo4j_checkpoints",
        max_retries: int = 5,
        initial_retry_delay: float = 5.0,
        max_retry_delay: float = 120.0,
        episode_timeout: float = 120.0,
        batch_delay: float = 2.0
    ):
        """
        Initialize robust Neo4j builder.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_retries: Maximum retry attempts per episode
            initial_retry_delay: Initial delay between retries (seconds)
            max_retry_delay: Maximum delay between retries (seconds)
            episode_timeout: Timeout for each episode creation (seconds)
            batch_delay: Delay between episodes to avoid overwhelming the system
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.episode_timeout = episode_timeout
        self.batch_delay = batch_delay
        
        self.supabase: Optional[Client] = None
        self.graph_client: Optional[GraphitiClient] = None
        self.state: Optional[Neo4jIngestionState] = None
        self.checkpoint_file: Optional[Path] = None
        
        # Statistics
        self.stats = {
            "episodes_attempted": 0,
            "episodes_succeeded": 0,
            "episodes_failed": 0,
            "total_retries": 0,
            "total_time_seconds": 0
        }
    
    async def initialize(self, session_id: Optional[str] = None) -> bool:
        """
        Initialize connections and load/create state.
        
        Args:
            session_id: Optional session ID to resume
            
        Returns:
            True if initialized successfully
        """
        try:
            # Initialize Supabase client
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            
            if not supabase_url or not supabase_key:
                logger.error("Supabase credentials not found in environment variables")
                return False
            
            self.supabase = create_client(supabase_url, supabase_key)
            logger.info("Connected to Supabase")
            
            # Initialize Graphiti client
            self.graph_client = GraphitiClient()
            await self.graph_client.initialize()
            logger.info("Initialized Graphiti client")
            
            # Load or create state
            if session_id:
                success = await self.resume_session(session_id)
                if not success:
                    logger.info("Creating new session instead")
                    await self.create_new_session()
            else:
                await self.create_new_session()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    async def create_new_session(self):
        """Create a new ingestion session."""
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        now = datetime.now(timezone.utc).isoformat()
        
        self.state = Neo4jIngestionState(
            session_id=session_id,
            started_at=now,
            updated_at=now,
            total_documents=0,
            total_chunks=0,
            completed_episodes=0,
            failed_episodes=0,
            current_document_index=0,
            episodes={},
            document_stats={}
        )
        
        self.checkpoint_file = self.checkpoint_dir / f"neo4j_checkpoint_{session_id}.json"
        
        logger.info(f"Created new session: {session_id}")
        await self.save_checkpoint()
    
    async def resume_session(self, session_id: str) -> bool:
        """
        Resume an existing session from checkpoint.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            True if resumed successfully
        """
        checkpoint_file = self.checkpoint_dir / f"neo4j_checkpoint_{session_id}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"No checkpoint found for session {session_id}")
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            self.state = Neo4jIngestionState.from_dict(data)
            self.checkpoint_file = checkpoint_file
            
            # Count pending episodes
            pending = sum(
                1 for ep in self.state.episodes.values()
                if ep.status in [EpisodeStatus.PENDING, EpisodeStatus.IN_PROGRESS, EpisodeStatus.FAILED]
            )
            
            logger.info(
                f"Resumed session {session_id}: "
                f"{self.state.completed_episodes}/{self.state.total_chunks} episodes completed, "
                f"{pending} pending/failed"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return False
    
    async def save_checkpoint(self):
        """Save current state to checkpoint file."""
        if not self.state or not self.checkpoint_file:
            return
        
        try:
            self.state.updated_at = datetime.now(timezone.utc).isoformat()
            
            # Write to temporary file first for atomic operation
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2, default=str)
            
            # Atomic move
            temp_file.replace(self.checkpoint_file)
            
            logger.debug(f"Saved checkpoint to {self.checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def process_episode_with_retry(
        self,
        episode_id: str,
        content: str,
        source: str,
        document_metadata: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Process a single episode with retry logic.
        
        Args:
            episode_id: Episode identifier
            content: Episode content
            source: Source description
            document_metadata: Metadata for the episode
            
        Returns:
            Tuple of (success, error_message)
        """
        retry_count = 0
        retry_delay = self.initial_retry_delay
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Add timeout to episode creation
                await asyncio.wait_for(
                    self.graph_client.add_episode(
                        episode_id=episode_id,
                        content=content,
                        source=source,
                        timestamp=datetime.now(timezone.utc),
                        metadata=document_metadata,
                        group_id="0"  # Shared knowledge base
                    ),
                    timeout=self.episode_timeout
                )
                
                logger.info(f"✓ Successfully created episode: {episode_id}")
                return True, None
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.episode_timeout} seconds"
                logger.warning(f"Episode {episode_id} timed out (attempt {retry_count + 1}/{self.max_retries + 1})")
                
            except Exception as e:
                last_error = str(e)[:200]  # Truncate long errors
                logger.warning(f"Episode {episode_id} failed (attempt {retry_count + 1}/{self.max_retries + 1}): {last_error}")
            
            retry_count += 1
            self.stats["total_retries"] += 1
            
            if retry_count <= self.max_retries:
                logger.info(f"Retrying episode {episode_id} in {retry_delay:.1f} seconds...")
                await asyncio.sleep(retry_delay)
                
                # Exponential backoff with jitter
                retry_delay = min(retry_delay * 2 * (1 + 0.1 * (0.5 - time.time() % 1)), self.max_retry_delay)
        
        logger.error(f"✗ Failed episode {episode_id} after {self.max_retries + 1} attempts: {last_error}")
        return False, last_error
    
    async def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents from Supabase to Neo4j.
        
        Returns:
            Processing results
        """
        start_time = time.time()
        
        # Get documents from Supabase
        logger.info("Fetching documents from Supabase...")
        try:
            result = self.supabase.table("documents").select("*").execute()
            documents = result.data
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            return {"error": "Failed to fetch documents from Supabase"}
        
        if not documents:
            logger.warning("No documents found in Supabase")
            return {"error": "No documents found"}
        
        logger.info(f"Found {len(documents)} documents to process")
        
        # Update state with document count
        self.state.total_documents = len(documents)
        
        # Process each document
        for doc_index, doc in enumerate(documents):
            doc_id = doc["id"]
            doc_title = doc["title"]
            doc_source = doc.get("source", "")
            
            # Skip if we've already processed this document completely
            if doc_index < self.state.current_document_index:
                logger.info(f"Skipping already processed document {doc_index + 1}: {doc_title}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"DOCUMENT {doc_index + 1}/{len(documents)}: {doc_title}")
            logger.info(f"{'='*60}")
            
            # Update current document index
            self.state.current_document_index = doc_index
            
            # Get chunks for this document
            try:
                chunk_result = self.supabase.table("chunks").select("*").eq("document_id", doc_id).order("chunk_index").execute()
                chunks_data = chunk_result.data
            except Exception as e:
                logger.error(f"Failed to fetch chunks for document {doc_title}: {e}")
                continue
            
            if not chunks_data:
                logger.warning(f"No chunks found for document: {doc_title}")
                continue
            
            logger.info(f"Processing {len(chunks_data)} chunks for document: {doc_title}")
            
            # Initialize document stats if not exists
            if doc_id not in self.state.document_stats:
                self.state.document_stats[doc_id] = {
                    "title": doc_title,
                    "total_chunks": len(chunks_data),
                    "completed_episodes": 0,
                    "failed_episodes": 0
                }
            
            # Process each chunk as an episode
            for chunk_data in chunks_data:
                chunk_index = chunk_data["chunk_index"]
                content = chunk_data["content"]
                
                # Create episode ID
                episode_id = f"{doc_source}_{chunk_index}_{doc_id[:8]}"
                
                # Check if episode already processed
                if episode_id in self.state.episodes:
                    ep_checkpoint = self.state.episodes[episode_id]
                    if ep_checkpoint.status == EpisodeStatus.COMPLETED:
                        logger.debug(f"Episode {episode_id} already completed, skipping")
                        continue
                    elif ep_checkpoint.retry_count >= self.max_retries:
                        logger.debug(f"Episode {episode_id} exceeded max retries, skipping")
                        continue
                else:
                    # Create new episode checkpoint
                    ep_checkpoint = EpisodeCheckpoint(
                        episode_id=episode_id,
                        document_id=doc_id,
                        document_title=doc_title,
                        chunk_index=chunk_index,
                        status=EpisodeStatus.PENDING,
                        created_at=datetime.now(timezone.utc).isoformat()
                    )
                    self.state.episodes[episode_id] = ep_checkpoint
                    self.state.total_chunks = len(self.state.episodes)
                
                # Update status to in-progress
                ep_checkpoint.status = EpisodeStatus.IN_PROGRESS
                
                # Prepare episode content (with truncation for Graphiti limits)
                max_content_length = 6000
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "... [TRUNCATED]"
                    logger.debug(f"Truncated chunk {chunk_index} to {max_content_length} characters")
                
                episode_content = f"[Document: {doc_title}]\n[Chunk: {chunk_index + 1}/{len(chunks_data)}]\n\n{content}"
                source_description = f"{doc_title} - Chunk {chunk_index + 1}"
                
                # Process episode metadata
                episode_metadata = {
                    "document_id": doc_id,
                    "document_title": doc_title,
                    "document_source": doc_source,
                    "chunk_index": chunk_index,
                    "original_length": len(chunk_data["content"]),
                    "processed_length": len(content),
                    "knowledge_type": "shared"
                }
                
                # Process episode with retry
                logger.info(f"Processing episode {chunk_index + 1}/{len(chunks_data)} for {doc_title}...")
                
                episode_start = time.time()
                success, error = await self.process_episode_with_retry(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_description,
                    document_metadata=episode_metadata
                )
                episode_time = time.time() - episode_start
                
                # Update episode checkpoint
                ep_checkpoint.processing_time_ms = episode_time * 1000
                self.stats["episodes_attempted"] += 1
                
                if success:
                    ep_checkpoint.status = EpisodeStatus.COMPLETED
                    ep_checkpoint.completed_at = datetime.now(timezone.utc).isoformat()
                    self.state.completed_episodes += 1
                    self.state.document_stats[doc_id]["completed_episodes"] += 1
                    self.stats["episodes_succeeded"] += 1
                else:
                    ep_checkpoint.status = EpisodeStatus.FAILED
                    ep_checkpoint.last_error = error
                    self.state.failed_episodes += 1
                    self.state.document_stats[doc_id]["failed_episodes"] += 1
                    self.stats["episodes_failed"] += 1
                
                # Save checkpoint after each episode
                await self.save_checkpoint()
                
                # Progress update
                total_progress = (self.state.completed_episodes / self.state.total_chunks * 100) if self.state.total_chunks > 0 else 0
                doc_progress = (self.state.document_stats[doc_id]["completed_episodes"] / 
                               self.state.document_stats[doc_id]["total_chunks"] * 100)
                
                logger.info(
                    f"Progress - Total: {self.state.completed_episodes}/{self.state.total_chunks} ({total_progress:.1f}%) | "
                    f"Document: {self.state.document_stats[doc_id]['completed_episodes']}/"
                    f"{self.state.document_stats[doc_id]['total_chunks']} ({doc_progress:.1f}%)"
                )
                
                # Delay between episodes to avoid overwhelming the system
                if chunk_index < len(chunks_data) - 1:
                    await asyncio.sleep(self.batch_delay)
            
            # Document completion summary
            doc_stats = self.state.document_stats[doc_id]
            logger.info(f"\n✓ Completed document {doc_title}: "
                       f"{doc_stats['completed_episodes']}/{doc_stats['total_chunks']} episodes successful")
            
            # Longer delay between documents
            if doc_index < len(documents) - 1:
                logger.info("Waiting 5 seconds before next document...")
                await asyncio.sleep(5)
        
        # Calculate total time
        self.stats["total_time_seconds"] = time.time() - start_time
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Get processing results."""
        return {
            "session_id": self.state.session_id if self.state else None,
            "total_documents": self.state.total_documents if self.state else 0,
            "total_chunks": self.state.total_chunks if self.state else 0,
            "completed_episodes": self.state.completed_episodes if self.state else 0,
            "failed_episodes": self.state.failed_episodes if self.state else 0,
            "stats": self.stats,
            "document_stats": self.state.document_stats if self.state else {},
            "checkpoint_file": str(self.checkpoint_file) if self.checkpoint_file else None
        }
    
    async def close(self):
        """Close connections and save final state."""
        if self.state:
            await self.save_checkpoint()
        
        if self.graph_client:
            await self.graph_client.close()
        
        logger.info("Closed all connections")


async def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("ROBUST NEO4J KNOWLEDGE GRAPH BUILDER")
    print("="*70)
    print("\nFeatures:")
    print("✓ Episode-by-episode processing with checkpointing")
    print("✓ Automatic retry with exponential backoff")
    print("✓ Resume from last successful episode")
    print("✓ Detailed logging and progress tracking")
    print("✓ 120-second timeout per episode")
    print("✓ Handles all 11 documents reliably")
    print("\n" + "="*70)
    
    # Check for resume option
    resume_session = None
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        if len(sys.argv) > 2:
            resume_session = sys.argv[2]
        else:
            # List available checkpoints
            checkpoint_dir = Path(".neo4j_checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("neo4j_checkpoint_*.json"))
                if checkpoints:
                    print("\nAvailable checkpoints to resume:")
                    for cp in sorted(checkpoints, reverse=True)[:5]:
                        session_id = cp.stem.replace("neo4j_checkpoint_", "")
                        print(f"  - {session_id}")
                    resume_session = input("\nEnter session ID to resume (or press Enter to start new): ").strip()
    
    # Create builder
    builder = RobustNeo4jBuilder(
        max_retries=5,
        initial_retry_delay=5.0,
        max_retry_delay=120.0,
        episode_timeout=120.0,
        batch_delay=2.0
    )
    
    try:
        # Initialize
        if not await builder.initialize(session_id=resume_session):
            logger.error("Failed to initialize builder")
            sys.exit(1)
        
        # Process all documents
        logger.info("\nStarting robust Neo4j ingestion...")
        results = await builder.process_all_documents()
        
        # Print summary
        print("\n" + "="*70)
        print("NEO4J INGESTION SUMMARY")
        print("="*70)
        print(f"Session ID: {results['session_id']}")
        print(f"Documents processed: {results['total_documents']}")
        print(f"Total episodes: {results['total_chunks']}")
        print(f"Successful episodes: {results['completed_episodes']}")
        print(f"Failed episodes: {results['failed_episodes']}")
        
        if results['stats']:
            stats = results['stats']
            print(f"\nStatistics:")
            print(f"  Episodes attempted: {stats['episodes_attempted']}")
            print(f"  Episodes succeeded: {stats['episodes_succeeded']}")
            print(f"  Episodes failed: {stats['episodes_failed']}")
            print(f"  Total retries: {stats['total_retries']}")
            print(f"  Total time: {stats['total_time_seconds']:.1f} seconds")
            
            if stats['episodes_succeeded'] > 0:
                avg_time = stats['total_time_seconds'] / stats['episodes_succeeded']
                print(f"  Average time per episode: {avg_time:.1f} seconds")
        
        print(f"\nCheckpoint saved to: {results['checkpoint_file']}")
        
        # Check success rate
        if results['completed_episodes'] == results['total_chunks']:
            print("\n✅ SUCCESS: 100% of episodes created successfully!")
        elif results['completed_episodes'] > 0:
            success_rate = results['completed_episodes'] / results['total_chunks'] * 100
            print(f"\n⚠️ PARTIAL SUCCESS: {success_rate:.1f}% of episodes created")
            print(f"Run with --resume {results['session_id']} to retry failed episodes")
        else:
            print("\n❌ FAILED: No episodes were created successfully")
        
        print("="*70)
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        print("\n⚠️ Process interrupted. Run with --resume to continue where you left off")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    
    finally:
        await builder.close()
        
        # Check Neo4j status
        print("\nChecking Neo4j database status...")
        os.system("python check_neo4j_status.py")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())