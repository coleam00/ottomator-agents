"""
Checkpoint Manager for Robust Document Ingestion
Provides document-level checkpointing, atomic saves with rollback, and resume capability.
"""

import os
import json
import logging
import shutil
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class DocumentStatus(Enum):
    """Status of document processing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class DocumentCheckpoint:
    """Checkpoint for a single document."""
    document_id: str
    file_path: str
    title: str
    status: DocumentStatus
    chunks_processed: int = 0
    total_chunks: int = 0
    episodes_created: int = 0
    entities_extracted: int = 0
    retry_count: int = 0
    last_error: Optional[str] = None
    processing_time_ms: float = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentCheckpoint':
        """Create from dictionary."""
        data['status'] = DocumentStatus(data['status'])
        return cls(**data)


@dataclass
class IngestionCheckpoint:
    """Overall ingestion checkpoint."""
    session_id: str
    started_at: str
    updated_at: str
    total_documents: int
    completed_documents: int
    failed_documents: int
    skipped_documents: int
    total_chunks_created: int
    total_episodes_created: int
    total_entities_extracted: int
    documents: Dict[str, DocumentCheckpoint] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['documents'] = {
            doc_id: checkpoint.to_dict() 
            for doc_id, checkpoint in self.documents.items()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IngestionCheckpoint':
        """Create from dictionary."""
        documents = {
            doc_id: DocumentCheckpoint.from_dict(doc_data)
            for doc_id, doc_data in data.get('documents', {}).items()
        }
        data['documents'] = documents
        return cls(**data)


class CheckpointManager:
    """Manages checkpointing for robust document ingestion."""
    
    def __init__(
        self,
        checkpoint_dir: str = ".ingestion_checkpoints",
        enable_atomic_saves: bool = True,
        auto_save_interval: float = 30.0  # seconds
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            enable_atomic_saves: Use atomic file operations
            auto_save_interval: Auto-save interval in seconds
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_atomic_saves = enable_atomic_saves
        self.auto_save_interval = auto_save_interval
        
        self.current_checkpoint: Optional[IngestionCheckpoint] = None
        self.checkpoint_file: Optional[Path] = None
        self._auto_save_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def initialize_session(
        self,
        session_id: str,
        total_documents: int,
        configuration: Dict[str, Any]
    ) -> IngestionCheckpoint:
        """
        Initialize a new ingestion session.
        
        Args:
            session_id: Unique session identifier
            total_documents: Total number of documents to process
            configuration: Ingestion configuration
            
        Returns:
            Initialized checkpoint
        """
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            
            self.current_checkpoint = IngestionCheckpoint(
                session_id=session_id,
                started_at=now,
                updated_at=now,
                total_documents=total_documents,
                completed_documents=0,
                failed_documents=0,
                skipped_documents=0,
                total_chunks_created=0,
                total_episodes_created=0,
                total_entities_extracted=0,
                documents={},
                configuration=configuration,
                errors=[]
            )
            
            self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{session_id}.json"
            
            # Save initial checkpoint
            await self._save_checkpoint()
            
            # Start auto-save task
            if self.auto_save_interval > 0:
                self._auto_save_task = asyncio.create_task(self._auto_save_loop())
            
            logger.info(f"Initialized checkpoint for session {session_id} with {total_documents} documents")
            
            return self.current_checkpoint
    
    async def resume_session(self, session_id: str) -> Optional[IngestionCheckpoint]:
        """
        Resume an existing session from checkpoint.
        
        Args:
            session_id: Session ID to resume
            
        Returns:
            Checkpoint if found, None otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{session_id}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"No checkpoint found for session {session_id}")
            return None
        
        try:
            async with self._lock:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                self.current_checkpoint = IngestionCheckpoint.from_dict(data)
                self.checkpoint_file = checkpoint_file
                
                # Restart auto-save task
                if self.auto_save_interval > 0:
                    self._auto_save_task = asyncio.create_task(self._auto_save_loop())
                
                # Log resume information
                pending = sum(
                    1 for doc in self.current_checkpoint.documents.values()
                    if doc.status in [DocumentStatus.PENDING, DocumentStatus.IN_PROGRESS]
                )
                
                logger.info(
                    f"Resumed session {session_id}: "
                    f"{self.current_checkpoint.completed_documents}/{self.current_checkpoint.total_documents} completed, "
                    f"{pending} pending"
                )
                
                return self.current_checkpoint
                
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return None
    
    async def start_document(
        self,
        document_id: str,
        file_path: str,
        title: str,
        checksum: Optional[str] = None
    ) -> DocumentCheckpoint:
        """
        Mark document as started.
        
        Args:
            document_id: Document identifier
            file_path: Path to document file
            title: Document title
            checksum: Optional file checksum
            
        Returns:
            Document checkpoint
        """
        if not self.current_checkpoint:
            raise RuntimeError("No active checkpoint session")
        
        async with self._lock:
            # Calculate checksum if not provided
            if not checksum and os.path.exists(file_path):
                checksum = self._calculate_checksum(file_path)
            
            # Check if document already exists
            if document_id in self.current_checkpoint.documents:
                doc_checkpoint = self.current_checkpoint.documents[document_id]
                
                # Skip if already completed
                if doc_checkpoint.status == DocumentStatus.COMPLETED:
                    logger.info(f"Document {document_id} already completed, skipping")
                    doc_checkpoint.status = DocumentStatus.SKIPPED
                    self.current_checkpoint.skipped_documents += 1
                    return doc_checkpoint
                
                # Retry if failed
                if doc_checkpoint.status == DocumentStatus.FAILED:
                    doc_checkpoint.status = DocumentStatus.RETRYING
                    doc_checkpoint.retry_count += 1
            else:
                # Create new document checkpoint
                doc_checkpoint = DocumentCheckpoint(
                    document_id=document_id,
                    file_path=file_path,
                    title=title,
                    status=DocumentStatus.IN_PROGRESS,
                    checksum=checksum,
                    started_at=datetime.now(timezone.utc).isoformat()
                )
                self.current_checkpoint.documents[document_id] = doc_checkpoint
            
            doc_checkpoint.status = DocumentStatus.IN_PROGRESS
            self.current_checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Started processing document {document_id}: {title}")
            
            return doc_checkpoint
    
    async def update_document_progress(
        self,
        document_id: str,
        chunks_processed: Optional[int] = None,
        total_chunks: Optional[int] = None,
        episodes_created: Optional[int] = None,
        entities_extracted: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update document processing progress.
        
        Args:
            document_id: Document identifier
            chunks_processed: Number of chunks processed
            total_chunks: Total number of chunks
            episodes_created: Number of episodes created
            entities_extracted: Number of entities extracted
            metadata: Additional metadata
        """
        if not self.current_checkpoint:
            return
        
        async with self._lock:
            if document_id not in self.current_checkpoint.documents:
                logger.warning(f"Document {document_id} not found in checkpoint")
                return
            
            doc_checkpoint = self.current_checkpoint.documents[document_id]
            
            if chunks_processed is not None:
                doc_checkpoint.chunks_processed = chunks_processed
            if total_chunks is not None:
                doc_checkpoint.total_chunks = total_chunks
            if episodes_created is not None:
                doc_checkpoint.episodes_created = episodes_created
            if entities_extracted is not None:
                doc_checkpoint.entities_extracted = entities_extracted
            if metadata:
                doc_checkpoint.metadata.update(metadata)
            
            self.current_checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
    
    async def complete_document(
        self,
        document_id: str,
        chunks_created: int,
        episodes_created: int,
        entities_extracted: int,
        processing_time_ms: float
    ):
        """
        Mark document as completed.
        
        Args:
            document_id: Document identifier
            chunks_created: Total chunks created
            episodes_created: Total episodes created
            entities_extracted: Total entities extracted
            processing_time_ms: Processing time in milliseconds
        """
        if not self.current_checkpoint:
            return
        
        async with self._lock:
            if document_id not in self.current_checkpoint.documents:
                logger.warning(f"Document {document_id} not found in checkpoint")
                return
            
            doc_checkpoint = self.current_checkpoint.documents[document_id]
            doc_checkpoint.status = DocumentStatus.COMPLETED
            doc_checkpoint.chunks_processed = chunks_created
            doc_checkpoint.total_chunks = chunks_created
            doc_checkpoint.episodes_created = episodes_created
            doc_checkpoint.entities_extracted = entities_extracted
            doc_checkpoint.processing_time_ms = processing_time_ms
            doc_checkpoint.completed_at = datetime.now(timezone.utc).isoformat()
            
            # Update global statistics
            self.current_checkpoint.completed_documents += 1
            self.current_checkpoint.total_chunks_created += chunks_created
            self.current_checkpoint.total_episodes_created += episodes_created
            self.current_checkpoint.total_entities_extracted += entities_extracted
            self.current_checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
            
            logger.info(
                f"Completed document {document_id}: "
                f"{chunks_created} chunks, {episodes_created} episodes, "
                f"{entities_extracted} entities in {processing_time_ms:.0f}ms"
            )
            
            # Save checkpoint after completion
            await self._save_checkpoint()
    
    async def fail_document(
        self,
        document_id: str,
        error: str,
        processing_time_ms: float
    ):
        """
        Mark document as failed.
        
        Args:
            document_id: Document identifier
            error: Error message
            processing_time_ms: Processing time in milliseconds
        """
        if not self.current_checkpoint:
            return
        
        async with self._lock:
            if document_id not in self.current_checkpoint.documents:
                logger.warning(f"Document {document_id} not found in checkpoint")
                return
            
            doc_checkpoint = self.current_checkpoint.documents[document_id]
            doc_checkpoint.status = DocumentStatus.FAILED
            doc_checkpoint.last_error = error
            doc_checkpoint.processing_time_ms = processing_time_ms
            
            # Update global statistics
            self.current_checkpoint.failed_documents += 1
            self.current_checkpoint.errors.append(f"{document_id}: {error}")
            self.current_checkpoint.updated_at = datetime.now(timezone.utc).isoformat()
            
            logger.error(f"Failed document {document_id}: {error}")
            
            # Save checkpoint after failure
            await self._save_checkpoint()
    
    async def get_pending_documents(self) -> List[str]:
        """
        Get list of pending document IDs.
        
        Returns:
            List of document IDs that need processing
        """
        if not self.current_checkpoint:
            return []
        
        async with self._lock:
            pending = []
            for doc_id, doc_checkpoint in self.current_checkpoint.documents.items():
                if doc_checkpoint.status in [DocumentStatus.PENDING, DocumentStatus.IN_PROGRESS, DocumentStatus.FAILED]:
                    pending.append(doc_id)
            
            return pending
    
    async def save_checkpoint(self):
        """Public method to save checkpoint."""
        await self._save_checkpoint()
    
    async def _save_checkpoint(self):
        """Save checkpoint to file with atomic operation."""
        if not self.current_checkpoint or not self.checkpoint_file:
            return
        
        try:
            checkpoint_data = self.current_checkpoint.to_dict()
            
            if self.enable_atomic_saves:
                # Write to temporary file first
                temp_file = self.checkpoint_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, default=str)
                
                # Atomic move
                temp_file.replace(self.checkpoint_file)
            else:
                # Direct write
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.debug(f"Saved checkpoint to {self.checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    async def _auto_save_loop(self):
        """Auto-save checkpoint periodically."""
        while True:
            try:
                await asyncio.sleep(self.auto_save_interval)
                await self._save_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save error: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def close(self):
        """Close checkpoint manager and save final state."""
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
        
        await self._save_checkpoint()
        
        if self.current_checkpoint:
            logger.info(
                f"Checkpoint manager closed. Final stats: "
                f"{self.current_checkpoint.completed_documents}/{self.current_checkpoint.total_documents} completed, "
                f"{self.current_checkpoint.failed_documents} failed"
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get checkpoint summary."""
        if not self.current_checkpoint:
            return {}
        
        return {
            "session_id": self.current_checkpoint.session_id,
            "total_documents": self.current_checkpoint.total_documents,
            "completed_documents": self.current_checkpoint.completed_documents,
            "failed_documents": self.current_checkpoint.failed_documents,
            "skipped_documents": self.current_checkpoint.skipped_documents,
            "total_chunks_created": self.current_checkpoint.total_chunks_created,
            "total_episodes_created": self.current_checkpoint.total_episodes_created,
            "total_entities_extracted": self.current_checkpoint.total_entities_extracted,
            "progress_percentage": (
                self.current_checkpoint.completed_documents / self.current_checkpoint.total_documents * 100
                if self.current_checkpoint.total_documents > 0 else 0
            ),
            "errors_count": len(self.current_checkpoint.errors)
        }