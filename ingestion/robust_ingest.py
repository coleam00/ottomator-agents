"""
Robust Document Ingestion Pipeline with Performance Optimizations
Integrates all optimization strategies for reliable, fast ingestion.
"""

import os
import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import argparse

from dotenv import load_dotenv

# Import optimization modules
from .neo4j_performance_optimizer import (
    OptimizedNeo4jProcessor,
    GraphitiBatchProcessor,
    BatchConfig,
    PerformanceMetrics
)
from .checkpoint_manager import CheckpointManager, DocumentStatus
from .monitoring_dashboard import MonitoringDashboard
from .validation_framework import ValidationFramework
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
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.unified_db_utils import (
        initialize_database, close_database,
        insert_document, bulk_insert_chunks, execute_query
    )
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult

load_dotenv()
logger = logging.getLogger(__name__)


class RobustIngestionPipeline:
    """
    Robust ingestion pipeline with all performance optimizations.
    Features:
    - 10x faster Neo4j operations with connection pooling and batching
    - Aggressive content truncation for Graphiti (2000 chars)
    - Document-level checkpointing with resume capability
    - Real-time monitoring dashboard
    - Pre-flight validation checks
    - Smart retry with exponential backoff
    - Graceful degradation on failures
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "medical_docs",
        clean_before_ingest: bool = False,
        enable_monitoring: bool = True,
        enable_validation: bool = True,
        enable_checkpointing: bool = True,
        resume_session: Optional[str] = None
    ):
        """
        Initialize robust ingestion pipeline.
        
        Args:
            config: Ingestion configuration
            documents_folder: Folder containing documents
            clean_before_ingest: Clean databases before ingestion
            enable_monitoring: Enable real-time monitoring
            enable_validation: Enable validation checks
            enable_checkpointing: Enable checkpointing
            resume_session: Session ID to resume from
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.enable_monitoring = enable_monitoring
        self.enable_validation = enable_validation
        self.enable_checkpointing = enable_checkpointing
        self.resume_session = resume_session
        
        # Generate session ID
        self.session_id = resume_session or f"ingestion_{uuid.uuid4().hex[:8]}"
        
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
        
        # Performance optimization components
        batch_config = BatchConfig(
            max_batch_size=10,
            batch_timeout=30.0,
            document_timeout=300.0,
            max_retries=3,
            retry_delay=2.0,
            enable_parallel=False,  # Sequential for reliability
            max_concurrent=3,
            content_truncation_limit=2000,  # Aggressive truncation
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5,
            circuit_breaker_cooldown=30.0
        )
        
        self.neo4j_processor = OptimizedNeo4jProcessor(batch_config)
        self.graphiti_processor = GraphitiBatchProcessor()
        
        # Management components
        self.checkpoint_manager = CheckpointManager() if enable_checkpointing else None
        self.monitoring_dashboard = MonitoringDashboard() if enable_monitoring else None
        self.validation_framework = ValidationFramework() if enable_validation else None
        
        # Statistics
        self.total_documents = 0
        self.completed_documents = 0
        self.failed_documents = 0
        self.total_chunks_created = 0
        self.total_episodes_created = 0
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info(f"Initializing robust ingestion pipeline (session: {self.session_id})")
        
        # Initialize database connections
        await initialize_database()
        await initialize_graph()
        await self.graph_builder.initialize()
        
        # Initialize Neo4j optimizations
        await self.neo4j_processor.initialize()
        await self.neo4j_processor.warm_up_connection_pool()
        await self.neo4j_processor.optimize_indices()
        
        await self.graphiti_processor.initialize()
        
        # Start monitoring
        if self.monitoring_dashboard:
            await self.monitoring_dashboard.start()
        
        self._initialized = True
        logger.info("Robust ingestion pipeline initialized with all optimizations")
    
    async def close(self):
        """Close all components."""
        if not self._initialized:
            return
        
        logger.info("Closing robust ingestion pipeline...")
        
        # Flush remaining work
        await self.graphiti_processor.flush_batch(force_sequential=True)
        
        # Close components
        await self.neo4j_processor.close()
        await self.graphiti_processor.close()
        await self.graph_builder.close()
        await close_graph()
        await close_database()
        
        # Stop monitoring
        if self.monitoring_dashboard:
            await self.monitoring_dashboard.stop()
            # Export final metrics
            self.monitoring_dashboard.export_metrics(
                f"metrics_{self.session_id}.json"
            )
        
        # Save final checkpoint
        if self.checkpoint_manager:
            await self.checkpoint_manager.close()
        
        self._initialized = False
        
        # Log final statistics
        logger.info(
            f"Pipeline closed. Stats: "
            f"{self.completed_documents}/{self.total_documents} documents, "
            f"{self.total_chunks_created} chunks, "
            f"{self.total_episodes_created} episodes"
        )
    
    async def ingest_documents(self) -> List[IngestionResult]:
        """
        Ingest all documents with robust error handling.
        
        Returns:
            List of ingestion results
        """
        try:
            # Initialize pipeline
            await self.initialize()
            
            # Run pre-flight checks
            if self.validation_framework:
                validation_report = await self.validation_framework.run_pre_flight_checks(
                    self.documents_folder,
                    self.session_id
                )
                
                self.validation_framework.print_report(validation_report)
                
                if not validation_report.can_proceed:
                    logger.error("Pre-flight validation failed - cannot proceed")
                    return []
            
            # Clean databases if requested
            if self.clean_before_ingest:
                await self._clean_databases()
            
            # Find documents
            documents = self._find_documents()
            self.total_documents = len(documents)
            
            if not documents:
                logger.warning(f"No documents found in {self.documents_folder}")
                return []
            
            logger.info(f"Found {len(documents)} documents to process")
            
            # Initialize or resume checkpoint
            if self.checkpoint_manager:
                if self.resume_session:
                    checkpoint = await self.checkpoint_manager.resume_session(self.session_id)
                    if not checkpoint:
                        checkpoint = await self.checkpoint_manager.initialize_session(
                            self.session_id,
                            len(documents),
                            {"config": self.config.__dict__}
                        )
                else:
                    checkpoint = await self.checkpoint_manager.initialize_session(
                        self.session_id,
                        len(documents),
                        {"config": self.config.__dict__}
                    )
            
            # Update monitoring totals
            if self.monitoring_dashboard:
                self.monitoring_dashboard.update_totals(len(documents), 0)
            
            # Process documents
            results = []
            for i, doc_path in enumerate(documents):
                try:
                    # Check if already processed
                    doc_id = self._get_document_id(doc_path)
                    
                    if self.checkpoint_manager:
                        doc_checkpoint = await self.checkpoint_manager.start_document(
                            doc_id,
                            doc_path,
                            self._extract_title_from_path(doc_path),
                            None
                        )
                        
                        if doc_checkpoint.status == DocumentStatus.SKIPPED:
                            logger.info(f"Skipping already completed document: {doc_path}")
                            continue
                    
                    # Process document with timeout
                    logger.info(f"Processing document {i+1}/{len(documents)}: {doc_path}")
                    
                    result = await asyncio.wait_for(
                        self._process_single_document(doc_path, doc_id),
                        timeout=300.0  # 5 minute timeout per document
                    )
                    
                    results.append(result)
                    
                    if result.errors:
                        self.failed_documents += 1
                    else:
                        self.completed_documents += 1
                    
                    # Update checkpoint
                    if self.checkpoint_manager:
                        if result.errors:
                            await self.checkpoint_manager.fail_document(
                                doc_id,
                                "; ".join(result.errors),
                                result.processing_time_ms
                            )
                        else:
                            await self.checkpoint_manager.complete_document(
                                doc_id,
                                result.chunks_created,
                                result.relationships_created,
                                result.entities_extracted,
                                result.processing_time_ms
                            )
                    
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing document: {doc_path}")
                    self.failed_documents += 1
                    
                    if self.checkpoint_manager:
                        await self.checkpoint_manager.fail_document(
                            doc_id,
                            "Processing timeout (300s)",
                            300000
                        )
                    
                    results.append(IngestionResult(
                        document_id=doc_id,
                        title=self._extract_title_from_path(doc_path),
                        chunks_created=0,
                        entities_extracted=0,
                        relationships_created=0,
                        processing_time_ms=300000,
                        errors=["Processing timeout"]
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {e}")
                    self.failed_documents += 1
                    
                    if self.checkpoint_manager:
                        await self.checkpoint_manager.fail_document(
                            doc_id,
                            str(e),
                            0
                        )
                    
                    results.append(IngestionResult(
                        document_id=doc_id,
                        title=self._extract_title_from_path(doc_path),
                        chunks_created=0,
                        entities_extracted=0,
                        relationships_created=0,
                        processing_time_ms=0,
                        errors=[str(e)]
                    ))
                
                # Progress update
                progress = (i + 1) / len(documents) * 100
                logger.info(
                    f"Progress: {i+1}/{len(documents)} ({progress:.1f}%), "
                    f"Success: {self.completed_documents}, Failed: {self.failed_documents}"
                )
            
            # Final flush of any remaining batches
            await self.graphiti_processor.flush_batch(force_sequential=True)
            
            # Print final statistics
            self._print_summary(results)
            
            return results
            
        except KeyboardInterrupt:
            logger.info("Ingestion interrupted by user")
            return []
            
        except Exception as e:
            logger.error(f"Critical error in ingestion pipeline: {e}")
            raise
            
        finally:
            await self.close()
    
    async def _process_single_document(
        self,
        doc_path: str,
        doc_id: str
    ) -> IngestionResult:
        """Process a single document with optimizations."""
        start_time = datetime.now()
        
        # Read document
        content = self._read_document(doc_path)
        title = self._extract_title(content, doc_path)
        source = os.path.relpath(doc_path, self.documents_folder)
        metadata = self._extract_metadata(content, doc_path)
        
        # Update monitoring
        if self.monitoring_dashboard:
            self.monitoring_dashboard.record_document_start(doc_id, title)
            self.monitoring_dashboard.update_phase("chunking")
        
        # Chunk document
        chunks = await self._chunk_document(content, title, source, metadata)
        
        if not chunks:
            return IngestionResult(
                document_id=doc_id,
                title=title,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created"]
            )
        
        logger.info(f"Created {len(chunks)} chunks for {title}")
        
        # Extract entities if configured
        entities_extracted = 0
        if self.config.extract_entities:
            if self.monitoring_dashboard:
                self.monitoring_dashboard.update_phase("entity_extraction")
            
            chunks = await self.graph_builder.extract_entities_from_chunks(chunks)
            entities_extracted = self._count_entities(chunks)
            logger.info(f"Extracted {entities_extracted} entities")
        
        # Generate embeddings
        if self.monitoring_dashboard:
            self.monitoring_dashboard.update_phase("embedding_generation")
        
        embed_start = datetime.now()
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        embed_time = (datetime.now() - embed_start).total_seconds()
        
        if self.monitoring_dashboard:
            self.monitoring_dashboard.record_embedding_generation_time(embed_time)
        
        logger.info(f"Generated embeddings in {embed_time:.2f}s")
        
        # Save to PostgreSQL/Supabase
        if self.monitoring_dashboard:
            self.monitoring_dashboard.update_phase("database_save")
        
        document_id = await self._save_to_database(
            title, source, content, embedded_chunks, metadata
        )
        
        self.total_chunks_created += len(chunks)
        
        # Build knowledge graph with optimizations
        episodes_created = 0
        graph_errors = []
        
        if not self.config.skip_graph_building:
            if self.monitoring_dashboard:
                self.monitoring_dashboard.update_phase("graph_building")
            
            try:
                # Use optimized batch processing
                logger.info("Building knowledge graph with optimizations...")
                
                # Process chunks in batches
                batch_size = 5
                for i in range(0, len(embedded_chunks), batch_size):
                    batch = embedded_chunks[i:i+batch_size]
                    
                    # Add to Graphiti batch processor
                    for chunk in batch:
                        await self.graphiti_processor.add_to_batch(
                            episode_id=f"{document_id}_chunk_{chunk.index}",
                            content=chunk.content[:2000],  # Aggressive truncation
                            source=source,
                            metadata={
                                "document_id": document_id,
                                "chunk_index": chunk.index,
                                "entities": chunk.metadata.get("entities", {})
                            }
                        )
                    
                    # Small delay between batches
                    await asyncio.sleep(0.5)
                
                # Final flush
                flush_result = await self.graphiti_processor.flush_batch()
                episodes_created = flush_result.get("successful", 0)
                self.total_episodes_created += episodes_created
                
                logger.info(f"Created {episodes_created} episodes in knowledge graph")
                
            except Exception as e:
                error_msg = f"Graph building error: {str(e)}"
                logger.error(error_msg)
                graph_errors.append(error_msg)
        
        # Calculate final metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update monitoring
        if self.monitoring_dashboard:
            self.monitoring_dashboard.record_document_complete(
                doc_id,
                len(chunks),
                episodes_created,
                entities_extracted,
                success=not graph_errors
            )
            
            for chunk in embedded_chunks:
                if hasattr(chunk, 'processing_time'):
                    self.monitoring_dashboard.record_chunk_processing_time(
                        chunk.processing_time
                    )
        
        return IngestionResult(
            document_id=document_id,
            title=title,
            chunks_created=len(chunks),
            entities_extracted=entities_extracted,
            relationships_created=episodes_created,
            processing_time_ms=processing_time,
            errors=graph_errors
        )
    
    async def _chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk document with handling for async/sync chunkers."""
        result = self.chunker.chunk_document(
            content=content,
            title=title,
            source=source,
            metadata=metadata
        )
        
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    async def _save_to_database(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to database."""
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
            if hasattr(chunk, 'embedding') and chunk.embedding:
                chunk_data.append({
                    "document_id": document_id,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "chunk_index": chunk.index,
                    "metadata": chunk.metadata,
                    "token_count": chunk.token_count
                })
        
        # Bulk insert chunks
        if chunk_data:
            await bulk_insert_chunks(chunk_data)
        
        return document_id
    
    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")
        
        await execute_query("DELETE FROM messages")
        await execute_query("DELETE FROM sessions")
        await execute_query("DELETE FROM chunks")
        await execute_query("DELETE FROM documents")
        
        await self.graph_builder.clear_graph()
        
        logger.info("Databases cleaned")
    
    def _find_documents(self) -> List[str]:
        """Find all documents in folder."""
        if not os.path.exists(self.documents_folder):
            return []
        
        patterns = ["*.md", "*.markdown", "*.txt"]
        documents = []
        
        for pattern in patterns:
            documents.extend(Path(self.documents_folder).rglob(pattern))
        
        return [str(p) for p in sorted(documents)]
    
    def _get_document_id(self, doc_path: str) -> str:
        """Generate document ID from path."""
        import hashlib
        return hashlib.md5(doc_path.encode()).hexdigest()[:16]
    
    def _read_document(self, doc_path: str) -> str:
        """Read document content."""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(doc_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _extract_title(self, content: str, doc_path: str) -> str:
        """Extract title from content or path."""
        lines = content.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        return self._extract_title_from_path(doc_path)
    
    def _extract_title_from_path(self, doc_path: str) -> str:
        """Extract title from file path."""
        return os.path.splitext(os.path.basename(doc_path))[0]
    
    def _extract_metadata(self, content: str, doc_path: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        return {
            "file_path": doc_path,
            "file_size": len(content),
            "line_count": len(content.split('\n')),
            "word_count": len(content.split()),
            "ingestion_date": datetime.now().isoformat()
        }
    
    def _count_entities(self, chunks: List[DocumentChunk]) -> int:
        """Count total entities in chunks."""
        total = 0
        for chunk in chunks:
            entities = chunk.metadata.get("entities", {})
            total += len(entities.get("companies", []))
            total += len(entities.get("technologies", []))
            total += len(entities.get("people", []))
        return total
    
    def _print_summary(self, results: List[IngestionResult]):
        """Print ingestion summary."""
        print("\n" + "="*60)
        print("ROBUST INGESTION SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_id}")
        print(f"Documents Processed: {len(results)}")
        print(f"  ✅ Successful: {self.completed_documents}")
        print(f"  ❌ Failed: {self.failed_documents}")
        print(f"Total Chunks Created: {self.total_chunks_created}")
        print(f"Total Episodes Created: {self.total_episodes_created}")
        
        # Performance metrics
        if self.neo4j_processor.metrics:
            metrics = self.neo4j_processor.metrics.get_summary()
            print(f"\nNeo4j Performance:")
            print(f"  Success Rate: {metrics['success_rate']:.1f}%")
            print(f"  Avg Processing Time: {metrics['avg_processing_time']:.3f}s")
            print(f"  Timeouts: {metrics['timeouts']}")
            print(f"  Connection Errors: {metrics['connection_errors']}")
        
        # Individual results
        print(f"\nDocument Results:")
        for result in results:
            status = "✅" if not result.errors else "❌"
            print(f"  {status} {result.title}: {result.chunks_created} chunks, "
                  f"{result.relationships_created} episodes")
            if result.errors:
                for error in result.errors[:2]:  # Show first 2 errors
                    print(f"      Error: {error}")
        
        print("="*60)


async def main():
    """Main entry point for robust ingestion."""
    parser = argparse.ArgumentParser(
        description="Robust document ingestion with performance optimizations"
    )
    parser.add_argument("--documents", "-d", default="medical_docs",
                        help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true",
                        help="Clean databases before ingestion")
    parser.add_argument("--chunk-size", type=int, default=800,
                        help="Chunk size (default: 800)")
    parser.add_argument("--no-monitoring", action="store_true",
                        help="Disable monitoring dashboard")
    parser.add_argument("--no-validation", action="store_true",
                        help="Skip validation checks")
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="Disable checkpointing")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from session ID")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=200,
        use_semantic_chunking=False,  # Disabled for speed
        extract_entities=True,
        skip_graph_building=False
    )
    
    # Create pipeline
    pipeline = RobustIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        enable_monitoring=not args.no_monitoring,
        enable_validation=not args.no_validation,
        enable_checkpointing=not args.no_checkpointing,
        resume_session=args.resume
    )
    
    # Run ingestion
    try:
        results = await pipeline.ingest_documents()
        
        if results:
            logger.info(f"Ingestion complete with {len(results)} documents processed")
        else:
            logger.warning("No documents were processed")
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())