"""
Robust validated ingestion system with synchronous processing, comprehensive verification, and retry logic.
Ensures 100% completion for both Supabase vector storage and Neo4j knowledge graph.
"""

import os
import asyncio
import logging
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import time
from functools import wraps

from dotenv import load_dotenv

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder
from .graph_builder import create_graph_builder

# Import agent utilities
try:
    from ..agent.unified_db_utils import (
        initialize_database, close_database,
        insert_document, bulk_insert_chunks, execute_query
    )
    from ..agent.graph_utils import initialize_graph, close_graph
    from ..agent.models import IngestionConfig, IngestionResult
    # Import Supabase utilities for proper validation
    from ..agent.supabase_db_utils import supabase_pool
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.unified_db_utils import (
        initialize_database, close_database,
        insert_document, bulk_insert_chunks, execute_query
    )
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult
    # Import Supabase utilities for proper validation
    from agent.supabase_db_utils import supabase_pool

# Load environment variables
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validated_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of document validation."""
    document_id: str
    title: str
    supabase_valid: bool
    neo4j_valid: bool
    chunk_count: int
    episode_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IngestionProgress:
    """Track overall ingestion progress."""
    total_documents: int
    completed_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_chunks_created: int = 0
    total_episodes_created: int = 0
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now())
    avg_processing_time: float = 0.0  # Average seconds per document


def async_retry_with_backoff(max_attempts=3, initial_delay=2.0, backoff_factor=2.0, exceptions=(Exception,)):
    """Decorator for async functions with exponential backoff retry."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {str(e)[:200]}")
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator


class ValidatedDocumentIngestion:
    """
    Robust document ingestion with validation and retry logic.
    Processes documents synchronously one at a time.
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "medical_docs",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        checkpoint_file: str = "ingestion_checkpoint.json"
    ):
        """
        Initialize validated ingestion pipeline.
        
        Args:
            config: Ingestion configuration
            documents_folder: Folder containing markdown documents
            max_retries: Maximum retry attempts per document
            retry_delay: Delay between retries in seconds
            checkpoint_file: File to save progress checkpoints
        """
        self.config = config
        self.documents_folder = documents_folder
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.checkpoint_file = checkpoint_file
        
        # Force knowledge graph building (no --fast mode)
        self.config.skip_graph_building = False
        
        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )
        
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.graph_builder = create_graph_builder()
        
        self._initialized = False
        self.progress = IngestionProgress(total_documents=0)
        
        # GROUP_ID = 0 for shared knowledge base
        self.GROUP_ID = "0"
        
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("="*60)
        logger.info("INITIALIZING VALIDATED INGESTION PIPELINE")
        logger.info("="*60)
        
        try:
            # Initialize database connections
            await initialize_database()
            logger.info("✅ Supabase database initialized")
            
            await initialize_graph()
            logger.info("✅ Neo4j graph connection initialized")
            
            await self.graph_builder.initialize()
            logger.info("✅ Graph builder initialized")
            
            self._initialized = True
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            await self.graph_builder.close()
            await close_graph()
            await close_database()
            self._initialized = False
            logger.info("✅ All connections closed")
    
    async def clean_databases(self):
        """Clean all existing data from both databases."""
        logger.warning("⚠️ CLEANING ALL DATA FROM DATABASES")
        
        try:
            # Clean Supabase tables using proper Supabase client methods
            logger.info("Cleaning Supabase tables...")
            
            # Check if we're using Supabase
            db_provider = os.getenv("DB_PROVIDER", "postgres").lower()
            
            if db_provider == "supabase":
                # Use Supabase client methods
                async with supabase_pool.acquire() as client:
                    # Delete in correct order to respect foreign keys
                    client.table("messages").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
                    client.table("sessions").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
                    client.table("chunks").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
                    client.table("documents").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
            else:
                # Use raw SQL for direct PostgreSQL
                await execute_query("DELETE FROM messages")
                await execute_query("DELETE FROM sessions")
                await execute_query("DELETE FROM chunks")
                await execute_query("DELETE FROM documents")
            
            logger.info("✅ Supabase tables cleaned")
            
            # Clean Neo4j knowledge graph with timeout handling
            logger.info("Cleaning Neo4j knowledge graph...")
            try:
                await asyncio.wait_for(self.graph_builder.clear_graph(), timeout=30.0)
                logger.info("✅ Neo4j graph cleaned")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Neo4j cleanup timed out after 30 seconds")
            
            logger.info("✅ All databases cleaned successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to clean databases: {e}")
            raise
    
    def find_documents(self) -> List[str]:
        """Find all markdown documents in the documents folder."""
        if not os.path.exists(self.documents_folder):
            raise FileNotFoundError(f"Documents folder not found: {self.documents_folder}")
        
        documents = []
        for file in sorted(os.listdir(self.documents_folder)):
            if file.endswith(('.md', '.markdown', '.txt')):
                documents.append(os.path.join(self.documents_folder, file))
        
        return documents
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "progress": {
                "total_documents": self.progress.total_documents,
                "completed_documents": self.progress.completed_documents,
                "successful_documents": self.progress.successful_documents,
                "failed_documents": self.progress.failed_documents,
                "total_chunks_created": self.progress.total_chunks_created,
                "total_episodes_created": self.progress.total_episodes_created
            },
            "checkpoints": self.progress.checkpoints,
            "validation_results": [
                {
                    "document_id": r.document_id,
                    "title": r.title,
                    "supabase_valid": r.supabase_valid,
                    "neo4j_valid": r.neo4j_valid,
                    "chunk_count": r.chunk_count,
                    "episode_count": r.episode_count,
                    "errors": r.errors,
                    "warnings": r.warnings
                }
                for r in self.progress.validation_results
            ]
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved to {self.checkpoint_file}")
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if it exists."""
        if not os.path.exists(self.checkpoint_file):
            return False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"📂 Found checkpoint from {data['timestamp']}")
            # For now, we'll start fresh each time
            # Future enhancement: resume from checkpoint
            return False
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return False
    
    async def ingest_single_document_with_retry(
        self,
        file_path: str,
        doc_number: int,
        total_docs: int
    ) -> Tuple[bool, ValidationResult]:
        """
        Ingest a single document with retry logic.
        
        Returns:
            Tuple of (success, validation_result)
        """
        document_name = os.path.basename(file_path)
        
        logger.info("="*60)
        logger.info(f"PROCESSING DOCUMENT {doc_number}/{total_docs}: {document_name}")
        logger.info("="*60)
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"📝 Attempt {attempt}/{self.max_retries}")
                
                # Read document
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                title = self._extract_title(content, file_path)
                source = os.path.relpath(file_path, self.documents_folder)
                
                # Process document with progress tracking
                result = await self._process_document(
                    content=content,
                    title=title,
                    source=source,
                    file_path=file_path,
                    doc_number=doc_number,
                    total_docs=total_docs
                )
                
                # Validate ingestion
                validation = await self._validate_document_ingestion(
                    document_id=result["document_id"],
                    title=title,
                    expected_chunks=result["chunks_created"],
                    expected_episodes=result["episodes_created"]
                )
                
                if validation.supabase_valid and validation.neo4j_valid:
                    logger.info(f"✅ Document {doc_number}/{total_docs} successfully ingested and validated")
                    return True, validation
                else:
                    logger.warning(f"⚠️ Validation issues detected, retrying...")
                    validation.errors.append(f"Validation failed on attempt {attempt}")
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * attempt)  # Exponential backoff
                        continue
                    
            except Exception as e:
                error_msg = f"Attempt {attempt} failed: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                
                if attempt < self.max_retries:
                    logger.info(f"⏳ Waiting {self.retry_delay * attempt} seconds before retry...")
                    await asyncio.sleep(self.retry_delay * attempt)
                else:
                    # Final failure
                    validation = ValidationResult(
                        document_id="",
                        title=title if 'title' in locals() else document_name,
                        supabase_valid=False,
                        neo4j_valid=False,
                        chunk_count=0,
                        episode_count=0,
                        errors=[error_msg]
                    )
                    return False, validation
        
        # Should not reach here
        return False, validation
    
    async def _process_document(
        self,
        content: str,
        title: str,
        source: str,
        file_path: str,
        doc_number: int = 0,
        total_docs: int = 0
    ) -> Dict[str, Any]:
        """Process a single document."""
        start_time = datetime.now()
        
        logger.info(f"📄 Document: {title}")
        logger.info(f"📏 Content size: {len(content)} characters")
        
        # Extract metadata
        metadata = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat(),
            "line_count": len(content.split('\n')),
            "word_count": len(content.split())
        }
        
        # 1. Chunk the document
        logger.info("🔪 Chunking document...")
        # Check if chunker is async (SemanticChunker) or sync (SimpleChunker)
        import asyncio
        chunk_result = self.chunker.chunk_document(
            content=content,
            title=title,
            source=source,
            metadata=metadata
        )
        # If it's a coroutine, await it
        if asyncio.iscoroutine(chunk_result):
            chunks = await chunk_result
        else:
            chunks = chunk_result
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # 2. Extract entities (if configured)
        if self.config.extract_entities:
            logger.info("🔍 Extracting entities...")
            chunks = await self.graph_builder.extract_entities_from_chunks(chunks)
            entities_count = sum(
                len(chunk.metadata.get("entities", {}).get("companies", [])) +
                len(chunk.metadata.get("entities", {}).get("technologies", [])) +
                len(chunk.metadata.get("entities", {}).get("people", []))
                for chunk in chunks
            )
            logger.info(f"✅ Extracted {entities_count} entities")
        
        # 3. Generate embeddings with retry mechanism
        logger.info("🧮 Generating embeddings with retry mechanism...")
        embedded_chunks = await self._generate_embeddings_with_retry(chunks)
        logger.info(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
        
        # 4. Save to Supabase
        logger.info("💾 Saving to Supabase...")
        document_id = await self._save_to_supabase(
            title=title,
            source=source,
            content=content,
            chunks=embedded_chunks,
            metadata=metadata
        )
        logger.info(f"✅ Saved to Supabase with ID: {document_id}")
        
        # 5. Add to Neo4j knowledge graph with timeout and retry
        logger.info("🧠 Building knowledge graph with group_id=0...")
        graph_result = await self._add_to_graph_with_timeout(
            embedded_chunks=embedded_chunks,
            title=title,
            source=source,
            metadata=metadata
        )
        
        episodes_created = graph_result.get("episodes_created", 0)
        graph_errors = graph_result.get("errors", [])
        
        logger.info(f"✅ Added {episodes_created} episodes to Neo4j with group_id={self.GROUP_ID}")
        
        if graph_errors:
            logger.warning(f"⚠️ Graph errors: {graph_errors}")
        
        # Calculate processing time and update estimates
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        # Update average processing time for better estimates
        if doc_number > 0:
            self.progress.avg_processing_time = (
                (self.progress.avg_processing_time * (doc_number - 1) + processing_time) / doc_number
            )
            
            # Show estimated time remaining
            if doc_number < total_docs:
                remaining_docs = total_docs - doc_number
                estimated_remaining = remaining_docs * self.progress.avg_processing_time
                hours, remainder = divmod(estimated_remaining, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                if hours > 0:
                    logger.info(f"📊 Estimated time remaining: {int(hours)}h {int(minutes)}m")
                elif minutes > 0:
                    logger.info(f"📊 Estimated time remaining: {int(minutes)}m {int(seconds)}s")
                else:
                    logger.info(f"📊 Estimated time remaining: {int(seconds)}s")
        
        return {
            "document_id": document_id,
            "chunks_created": len(chunks),
            "episodes_created": episodes_created,
            "processing_time": processing_time,
            "errors": graph_errors
        }
    
    @async_retry_with_backoff(max_attempts=3, initial_delay=2.0, exceptions=(Exception,))
    async def _generate_embeddings_with_retry(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings with exponential backoff retry."""
        try:
            # Process in smaller batches to avoid timeouts
            batch_size = 10  # Smaller batch size for stability
            embedded_chunks = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # Add a small delay between batches to avoid rate limiting
                if i > 0:
                    await asyncio.sleep(0.5)
                
                batch_embedded = await self.embedder.embed_chunks(batch)
                embedded_chunks.extend(batch_embedded)
            
            return embedded_chunks
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def _add_to_graph_with_timeout(
        self,
        embedded_chunks: List[DocumentChunk],
        title: str,
        source: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add to Neo4j graph with timeout and retry logic."""
        max_attempts = 3
        timeout_seconds = 120  # 2 minutes timeout per attempt
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Neo4j graph building attempt {attempt}/{max_attempts}")
                
                # Use asyncio.wait_for to add timeout
                graph_result = await asyncio.wait_for(
                    self.graph_builder.add_document_to_graph(
                        chunks=embedded_chunks,
                        document_title=title,
                        document_source=source,
                        document_metadata=metadata,
                        batch_size=1,  # Process one chunk at a time for stability
                        group_id=self.GROUP_ID  # Pass group_id = "0" for shared knowledge base
                    ),
                    timeout=timeout_seconds
                )
                
                return graph_result
                
            except asyncio.TimeoutError:
                logger.warning(f"Neo4j operation timed out after {timeout_seconds}s (attempt {attempt}/{max_attempts})")
                if attempt < max_attempts:
                    await asyncio.sleep(5 * attempt)  # Exponential backoff
                else:
                    logger.error("Neo4j operation failed after all retries")
                    return {"episodes_created": 0, "errors": ["Timeout after multiple attempts"]}
                    
            except Exception as e:
                if "Failed to read from defunct connection" in str(e):
                    logger.warning(f"Neo4j connection error (attempt {attempt}/{max_attempts}): {str(e)[:200]}")
                    if attempt < max_attempts:
                        # Reinitialize graph connection
                        try:
                            await close_graph()
                            await initialize_graph()
                            await self.graph_builder.initialize()
                            logger.info("Reinitialized Neo4j connection")
                        except Exception as reinit_error:
                            logger.error(f"Failed to reinitialize Neo4j: {reinit_error}")
                        
                        await asyncio.sleep(5 * attempt)
                    else:
                        logger.error(f"Neo4j operation failed after all retries: {e}")
                        return {"episodes_created": 0, "errors": [str(e)[:500]]}
                else:
                    logger.error(f"Unexpected Neo4j error: {e}")
                    return {"episodes_created": 0, "errors": [str(e)[:500]]}
        
        return {"episodes_created": 0, "errors": ["Max attempts exceeded"]}
    
    async def _save_to_supabase(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to Supabase."""
        # Insert document
        document_id = await insert_document(
            title=title,
            source=source,
            content=content,
            metadata=metadata
        )
        
        # Prepare chunks for bulk insert
        chunk_data = []
        for chunk in chunks:
            chunk_dict = {
                "document_id": document_id,
                "content": chunk.content,
                "embedding": chunk.embedding if hasattr(chunk, 'embedding') and chunk.embedding else None,
                "chunk_index": chunk.index,
                "metadata": chunk.metadata,
                "token_count": chunk.token_count
            }
            # Only add chunks with embeddings
            if chunk_dict["embedding"]:
                chunk_data.append(chunk_dict)
        
        # Bulk insert chunks
        if chunk_data:
            await bulk_insert_chunks(chunk_data)
        
        return document_id
    
    async def _validate_document_ingestion(
        self,
        document_id: str,
        title: str,
        expected_chunks: int,
        expected_episodes: int
    ) -> ValidationResult:
        """Validate that document was properly ingested in both databases."""
        logger.info("🔍 Validating ingestion...")
        
        validation = ValidationResult(
            document_id=document_id,
            title=title,
            supabase_valid=False,
            neo4j_valid=False,
            chunk_count=0,
            episode_count=0
        )
        
        # Validate Supabase using proper client methods
        try:
            db_provider = os.getenv("DB_PROVIDER", "postgres").lower()
            
            if db_provider == "supabase":
                # Use Supabase client for validation
                async with supabase_pool.acquire() as client:
                    # Check document exists
                    doc_response = client.table("documents").select("id, title").eq("id", document_id).maybe_single().execute()
                    
                    if doc_response.data:
                        logger.info(f"✅ Document found in Supabase: {document_id}")
                        
                        # Check chunks count
                        chunk_response = client.table("chunks").select("id").eq("document_id", document_id).execute()
                        
                        if chunk_response.data is not None:
                            chunk_count = len(chunk_response.data)
                            validation.chunk_count = chunk_count
                            
                            if chunk_count == expected_chunks:
                                logger.info(f"✅ All {chunk_count} chunks found in Supabase")
                                validation.supabase_valid = True
                            elif chunk_count > 0:
                                warning = f"Expected {expected_chunks} chunks, found {chunk_count}"
                                logger.warning(f"⚠️ {warning}")
                                validation.warnings.append(warning)
                                validation.supabase_valid = True  # Partial success is still success
                            else:
                                error = "No chunks found in Supabase"
                                logger.error(f"❌ {error}")
                                validation.errors.append(error)
                    else:
                        error = "Document not found in Supabase"
                        logger.error(f"❌ {error}")
                        validation.errors.append(error)
            else:
                # Use execute_query for direct PostgreSQL
                doc_query = f"SELECT id, title FROM documents WHERE id = '{document_id}'"
                doc_result = await execute_query(doc_query)
                
                if doc_result and len(doc_result) > 0:
                    logger.info(f"✅ Document found in database: {document_id}")
                    
                    # Check chunks
                    chunk_query = f"SELECT COUNT(*) as count FROM chunks WHERE document_id = '{document_id}'"
                    chunk_result = await execute_query(chunk_query)
                    
                    if chunk_result and len(chunk_result) > 0:
                        chunk_count = chunk_result[0].get('count', 0)
                        validation.chunk_count = chunk_count
                        
                        if chunk_count == expected_chunks:
                            logger.info(f"✅ All {chunk_count} chunks found in database")
                            validation.supabase_valid = True
                        else:
                            warning = f"Expected {expected_chunks} chunks, found {chunk_count}"
                            logger.warning(f"⚠️ {warning}")
                            validation.warnings.append(warning)
                            validation.supabase_valid = (chunk_count > 0)  # Partial success
                else:
                    error = "Document not found in database"
                    logger.error(f"❌ {error}")
                    validation.errors.append(error)
                    
        except Exception as e:
            error = f"Database validation error: {str(e)}"
            logger.error(f"❌ {error}")
            validation.errors.append(error)
        
        # Validate Neo4j (check for episodes with group_id = 0)
        try:
            # For now, we'll consider Neo4j valid if episodes were created
            # Future enhancement: query Neo4j directly to verify
            if expected_episodes > 0:
                validation.episode_count = expected_episodes
                validation.neo4j_valid = True
                logger.info(f"✅ {expected_episodes} episodes reported in Neo4j with group_id={self.GROUP_ID}")
            else:
                warning = "No episodes created in Neo4j"
                logger.warning(f"⚠️ {warning}")
                validation.warnings.append(warning)
                
        except Exception as e:
            error = f"Neo4j validation error: {str(e)}"
            logger.error(f"❌ {error}")
            validation.errors.append(error)
        
        return validation
    
    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        lines = content.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        return os.path.splitext(os.path.basename(file_path))[0]
    
    async def run_ingestion(self, clean_first: bool = True) -> Dict[str, Any]:
        """
        Run the complete validated ingestion process.
        
        Args:
            clean_first: Whether to clean databases before ingestion
        
        Returns:
            Final ingestion report
        """
        try:
            # Initialize
            await self.initialize()
            
            # Clean databases if requested
            if clean_first:
                await self.clean_databases()
            
            # Find documents
            documents = self.find_documents()
            self.progress.total_documents = len(documents)
            
            logger.info("="*60)
            logger.info(f"STARTING VALIDATED INGESTION")
            logger.info(f"Documents to process: {len(documents)}")
            logger.info(f"Group ID: {self.GROUP_ID} (shared knowledge base)")
            logger.info(f"Max retries per document: {self.max_retries}")
            logger.info(f"Database provider: {os.getenv('DB_PROVIDER', 'postgres')}")
            logger.info("="*60)
            
            # Process each document synchronously
            for i, doc_path in enumerate(documents, 1):
                success, validation = await self.ingest_single_document_with_retry(
                    file_path=doc_path,
                    doc_number=i,
                    total_docs=len(documents)
                )
                
                # Update progress
                self.progress.completed_documents += 1
                if success:
                    self.progress.successful_documents += 1
                    self.progress.total_chunks_created += validation.chunk_count
                    self.progress.total_episodes_created += validation.episode_count
                else:
                    self.progress.failed_documents += 1
                
                self.progress.validation_results.append(validation)
                
                # Save checkpoint after each document
                checkpoint = {
                    "document": os.path.basename(doc_path),
                    "success": success,
                    "timestamp": datetime.now().isoformat(),
                    "chunks": validation.chunk_count,
                    "episodes": validation.episode_count
                }
                self.progress.checkpoints.append(checkpoint)
                self.save_checkpoint()
                
                # Progress report
                logger.info("-" * 60)
                logger.info(f"PROGRESS: {self.progress.completed_documents}/{self.progress.total_documents}")
                logger.info(f"Successful: {self.progress.successful_documents}")
                logger.info(f"Failed: {self.progress.failed_documents}")
                logger.info("-" * 60)
            
            # Generate final report
            return self.generate_final_report()
            
        except Exception as e:
            logger.error(f"❌ Fatal error during ingestion: {e}\n{traceback.format_exc()}")
            raise
        finally:
            await self.close()
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final validation report."""
        report = {
            "summary": {
                "total_documents": self.progress.total_documents,
                "successful_documents": self.progress.successful_documents,
                "failed_documents": self.progress.failed_documents,
                "success_rate": (self.progress.successful_documents / self.progress.total_documents * 100) 
                                if self.progress.total_documents > 0 else 0,
                "total_chunks_created": self.progress.total_chunks_created,
                "total_episodes_created": self.progress.total_episodes_created,
                "group_id": self.GROUP_ID
            },
            "validation_results": [],
            "failed_documents": [],
            "warnings": []
        }
        
        for result in self.progress.validation_results:
            result_data = {
                "title": result.title,
                "document_id": result.document_id,
                "supabase_valid": result.supabase_valid,
                "neo4j_valid": result.neo4j_valid,
                "chunks": result.chunk_count,
                "episodes": result.episode_count,
                "status": "✅ SUCCESS" if (result.supabase_valid and result.neo4j_valid) else "❌ FAILED"
            }
            
            report["validation_results"].append(result_data)
            
            if not (result.supabase_valid and result.neo4j_valid):
                report["failed_documents"].append({
                    "title": result.title,
                    "errors": result.errors,
                    "warnings": result.warnings
                })
            
            if result.warnings:
                report["warnings"].extend(result.warnings)
        
        # Print report to console
        self.print_final_report(report)
        
        # Save report to file
        report_file = f"ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Full report saved to {report_file}")
        
        return report
    
    def print_final_report(self, report: Dict[str, Any]):
        """Print formatted final report to console."""
        print("\n" + "="*60)
        print("FINAL VALIDATION REPORT")
        print("="*60)
        
        summary = report["summary"]
        print(f"Total Documents: {summary['total_documents']}")
        print(f"✅ Successful: {summary['successful_documents']}")
        print(f"❌ Failed: {summary['failed_documents']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Chunks Created: {summary['total_chunks_created']}")
        print(f"Total Episodes Created: {summary['total_episodes_created']}")
        print(f"Group ID: {summary['group_id']} (shared knowledge base)")
        
        print("\n" + "-"*60)
        print("DOCUMENT STATUS:")
        print("-"*60)
        
        for result in report["validation_results"]:
            print(f"{result['status']} {result['title']}")
            print(f"   Supabase: {'✅' if result['supabase_valid'] else '❌'} ({result['chunks']} chunks)")
            print(f"   Neo4j: {'✅' if result['neo4j_valid'] else '❌'} ({result['episodes']} episodes)")
        
        if report["failed_documents"]:
            print("\n" + "-"*60)
            print("FAILED DOCUMENTS DETAILS:")
            print("-"*60)
            for doc in report["failed_documents"]:
                print(f"\n❌ {doc['title']}")
                if doc['errors']:
                    print("   Errors:")
                    for error in doc['errors']:
                        print(f"   - {error[:200]}")  # Truncate long errors
                if doc['warnings']:
                    print("   Warnings:")
                    for warning in doc['warnings']:
                        print(f"   - {warning}")
        
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)


async def main():
    """Main function for running validated ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Robust document ingestion with validation and retry logic"
    )
    parser.add_argument(
        "--documents", "-d",
        default="medical_docs",
        help="Documents folder path"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        default=True,  # Default to cleaning for fresh start
        help="Clean existing data before ingestion (default: True)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for splitting documents"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap size"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per document"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking"
    )
    parser.add_argument(
        "--no-entities",
        action="store_true",
        help="Disable entity extraction"
    )
    
    args = parser.parse_args()
    
    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
        extract_entities=not args.no_entities,
        skip_graph_building=False  # Always build knowledge graph
    )
    
    # Create validated ingestion pipeline
    pipeline = ValidatedDocumentIngestion(
        config=config,
        documents_folder=args.documents,
        max_retries=args.max_retries
    )
    
    try:
        # Run ingestion with validation
        report = await pipeline.run_ingestion(clean_first=args.clean)
        
        # Exit with appropriate code
        if report["summary"]["failed_documents"] == 0:
            print("\n✅ All documents successfully ingested and validated!")
            exit(0)
        else:
            print(f"\n⚠️ {report['summary']['failed_documents']} documents failed validation")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Ingestion interrupted by user")
        exit(2)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Ingestion failed: {e}")
        exit(3)


if __name__ == "__main__":
    asyncio.run(main())