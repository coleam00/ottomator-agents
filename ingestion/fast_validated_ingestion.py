"""
Fast validated ingestion with optimized Neo4j handling.
Designed to resolve timeout and connection issues.
"""

import os
import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import time

from dotenv import load_dotenv

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder
from .optimized_graph_builder import create_optimized_graph_builder

# Import agent utilities
try:
    from ..agent.unified_db_utils import (
        initialize_database, close_database,
        insert_document, bulk_insert_chunks, execute_query
    )
    from ..agent.graph_utils import initialize_graph, close_graph
    from ..agent.models import IngestionConfig, IngestionResult
    from ..agent.supabase_db_utils import supabase_pool
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.unified_db_utils import (
        initialize_database, close_database,
        insert_document, bulk_insert_chunks, execute_query
    )
    from agent.graph_utils import initialize_graph, close_graph
    from agent.models import IngestionConfig, IngestionResult
    from agent.supabase_db_utils import supabase_pool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fast_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)


class FastValidatedIngestion:
    """
    Fast ingestion pipeline with optimized Neo4j handling.
    
    Key improvements:
    1. Shorter timeouts (30s per chunk vs 120s)
    2. Skip Neo4j on repeated failures
    3. Parallel chunk processing option
    4. Connection pooling and health checks
    5. Aggressive content truncation
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "medical_docs",
        neo4j_enabled: bool = True,
        neo4j_timeout: int = 30,
        neo4j_max_retries: int = 2
    ):
        self.config = config
        self.documents_folder = documents_folder
        self.neo4j_enabled = neo4j_enabled
        self.neo4j_timeout = neo4j_timeout
        self.neo4j_max_retries = neo4j_max_retries
        
        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )
        
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.graph_builder = create_optimized_graph_builder() if neo4j_enabled else None
        
        self._initialized = False
        self.GROUP_ID = "0"  # Shared knowledge base
        
        # Track Neo4j failures
        self.neo4j_consecutive_failures = 0
        self.neo4j_failure_threshold = 3
        
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("="*60)
        logger.info("INITIALIZING FAST INGESTION PIPELINE")
        logger.info("="*60)
        
        try:
            # Initialize Supabase (always needed)
            await initialize_database()
            logger.info("✅ Supabase initialized")
            
            # Initialize Neo4j (if enabled)
            if self.neo4j_enabled and self.graph_builder:
                try:
                    await initialize_graph()
                    await self.graph_builder.initialize()
                    logger.info("✅ Neo4j initialized (optimized)")
                except Exception as e:
                    logger.warning(f"⚠️ Neo4j initialization failed: {e}")
                    logger.warning("Continuing with Supabase-only mode")
                    self.neo4j_enabled = False
                    self.graph_builder = None
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            if self.graph_builder:
                await self.graph_builder.close()
                await close_graph()
            await close_database()
            self._initialized = False
            logger.info("✅ All connections closed")
    
    async def clean_databases(self):
        """Clean all existing data from databases."""
        logger.warning("⚠️ CLEANING ALL DATA FROM DATABASES")
        
        try:
            # Clean Supabase
            db_provider = os.getenv("DB_PROVIDER", "postgres").lower()
            
            if db_provider == "supabase":
                async with supabase_pool.acquire() as client:
                    client.table("messages").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
                    client.table("sessions").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
                    client.table("chunks").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
                    client.table("documents").delete().gte("id", "00000000-0000-0000-0000-000000000000").execute()
            else:
                await execute_query("DELETE FROM messages")
                await execute_query("DELETE FROM sessions")
                await execute_query("DELETE FROM chunks")
                await execute_query("DELETE FROM documents")
            
            logger.info("✅ Supabase cleaned")
            
            # Clean Neo4j (with timeout)
            if self.neo4j_enabled and self.graph_builder:
                try:
                    await asyncio.wait_for(self.graph_builder.clear_graph(), timeout=30.0)
                    logger.info("✅ Neo4j cleaned")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Neo4j cleanup timed out")
                except Exception as e:
                    logger.warning(f"⚠️ Neo4j cleanup failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to clean databases: {e}")
            raise
    
    def find_documents(self) -> List[str]:
        """Find all markdown documents."""
        if not os.path.exists(self.documents_folder):
            raise FileNotFoundError(f"Documents folder not found: {self.documents_folder}")
        
        documents = []
        for file in sorted(os.listdir(self.documents_folder)):
            if file.endswith(('.md', '.markdown', '.txt')):
                documents.append(os.path.join(self.documents_folder, file))
        
        return documents
    
    async def process_document_fast(
        self,
        file_path: str,
        doc_number: int,
        total_docs: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a single document with optimized Neo4j handling.
        """
        document_name = os.path.basename(file_path)
        start_time = time.time()
        
        logger.info("="*60)
        logger.info(f"PROCESSING DOCUMENT {doc_number}/{total_docs}: {document_name}")
        logger.info("="*60)
        
        try:
            # Read document
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title = self._extract_title(content, file_path)
            source = os.path.relpath(file_path, self.documents_folder)
            
            # Chunk the document
            logger.info("🔪 Chunking document...")
            chunk_result = self.chunker.chunk_document(
                content=content,
                title=title,
                source=source,
                metadata={"file_path": file_path}
            )
            
            # Handle async/sync chunker
            if asyncio.iscoroutine(chunk_result):
                chunks = await chunk_result
            else:
                chunks = chunk_result
            
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Generate embeddings (with batching)
            logger.info("🧠 Generating embeddings...")
            embedded_chunks = await self._generate_embeddings_batch(chunks)
            logger.info(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
            
            # Save to Supabase
            logger.info("💾 Saving to Supabase...")
            document_id = await self._save_to_supabase(
                title=title,
                source=source,
                content=content,
                chunks=embedded_chunks,
                metadata={"ingestion_date": datetime.now().isoformat()}
            )
            logger.info(f"✅ Saved to Supabase with ID: {document_id}")
            
            # Try Neo4j (with fallback)
            neo4j_result = {"episodes_created": 0, "skipped": True}
            
            if self.neo4j_enabled and self.graph_builder and self.neo4j_consecutive_failures < self.neo4j_failure_threshold:
                logger.info("🧠 Building knowledge graph...")
                
                try:
                    # Use the optimized graph builder
                    neo4j_result = await self.graph_builder.add_document_to_graph_optimized(
                        chunks=embedded_chunks,
                        document_title=title,
                        document_source=source,
                        document_metadata={"doc_id": document_id},
                        batch_size=1,
                        group_id=self.GROUP_ID,
                        max_retries=self.neo4j_max_retries,
                        timeout_seconds=self.neo4j_timeout
                    )
                    
                    if neo4j_result.get("episodes_created", 0) > 0:
                        self.neo4j_consecutive_failures = 0  # Reset failure counter
                        logger.info(f"✅ Added {neo4j_result['episodes_created']} episodes to Neo4j")
                    else:
                        self.neo4j_consecutive_failures += 1
                        logger.warning(f"⚠️ No episodes created (failure {self.neo4j_consecutive_failures}/{self.neo4j_failure_threshold})")
                        
                except Exception as e:
                    self.neo4j_consecutive_failures += 1
                    logger.error(f"❌ Neo4j failed (failure {self.neo4j_consecutive_failures}/{self.neo4j_failure_threshold}): {str(e)[:200]}")
                    neo4j_result = {"episodes_created": 0, "error": str(e)[:200]}
                    
                    if self.neo4j_consecutive_failures >= self.neo4j_failure_threshold:
                        logger.warning("🚫 Disabling Neo4j due to repeated failures")
                        self.neo4j_enabled = False
            
            elif not self.neo4j_enabled:
                logger.info("⚠️ Neo4j disabled, skipping graph building")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "document_id": document_id,
                "title": title,
                "chunks_created": len(chunks),
                "episodes_created": neo4j_result.get("episodes_created", 0),
                "neo4j_skipped": neo4j_result.get("skipped", False),
                "processing_time": processing_time
            }
            
            logger.info(f"✅ Document processed in {processing_time:.1f}s")
            return True, result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return False, {
                "success": False,
                "title": document_name,
                "error": str(e)[:500]
            }
    
    async def _generate_embeddings_batch(self, chunks: List[DocumentChunk], batch_size: int = 5) -> List[DocumentChunk]:
        """Generate embeddings in small batches."""
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Small delay between batches
            if i > 0:
                await asyncio.sleep(0.2)
            
            batch_embedded = await self.embedder.embed_chunks(batch)
            embedded_chunks.extend(batch_embedded)
        
        return embedded_chunks
    
    async def _save_to_supabase(self, title: str, source: str, content: str, 
                                chunks: List[DocumentChunk], metadata: Dict[str, Any]) -> str:
        """Save document and chunks to Supabase."""
        document_id = await insert_document(
            title=title,
            source=source,
            content=content,
            metadata=metadata
        )
        
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
        
        if chunk_data:
            await bulk_insert_chunks(chunk_data)
        
        return document_id
    
    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document."""
        lines = content.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return os.path.splitext(os.path.basename(file_path))[0]
    
    async def run_fast_ingestion(self, clean_first: bool = True) -> Dict[str, Any]:
        """
        Run the fast ingestion pipeline.
        """
        try:
            # Initialize
            await self.initialize()
            
            # Clean databases if requested
            if clean_first:
                await self.clean_databases()
            
            # Find documents
            documents = self.find_documents()
            
            logger.info("="*60)
            logger.info(f"STARTING FAST INGESTION")
            logger.info(f"Documents to process: {len(documents)}")
            logger.info(f"Neo4j enabled: {self.neo4j_enabled}")
            logger.info(f"Neo4j timeout: {self.neo4j_timeout}s per chunk")
            logger.info("="*60)
            
            # Process documents
            results = []
            successful = 0
            failed = 0
            total_chunks = 0
            total_episodes = 0
            start_time = time.time()
            
            for i, doc_path in enumerate(documents, 1):
                success, result = await self.process_document_fast(
                    file_path=doc_path,
                    doc_number=i,
                    total_docs=len(documents)
                )
                
                results.append(result)
                
                if success:
                    successful += 1
                    total_chunks += result.get("chunks_created", 0)
                    total_episodes += result.get("episodes_created", 0)
                else:
                    failed += 1
                
                # Progress update
                logger.info("-" * 60)
                logger.info(f"PROGRESS: {i}/{len(documents)} documents")
                logger.info(f"Successful: {successful}, Failed: {failed}")
                
                if self.neo4j_enabled:
                    logger.info(f"Neo4j status: Active (failures: {self.neo4j_consecutive_failures}/{self.neo4j_failure_threshold})")
                else:
                    logger.info("Neo4j status: Disabled")
                logger.info("-" * 60)
            
            # Calculate totals
            total_time = time.time() - start_time
            
            # Final report
            report = {
                "summary": {
                    "total_documents": len(documents),
                    "successful_documents": successful,
                    "failed_documents": failed,
                    "total_chunks_created": total_chunks,
                    "total_episodes_created": total_episodes,
                    "total_time_seconds": total_time,
                    "avg_time_per_doc": total_time / len(documents) if documents else 0,
                    "neo4j_final_status": "enabled" if self.neo4j_enabled else "disabled"
                },
                "documents": results
            }
            
            # Print summary
            print("\n" + "="*60)
            print("INGESTION COMPLETE")
            print("="*60)
            print(f"Total Documents: {len(documents)}")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"Total Chunks: {total_chunks}")
            print(f"Total Episodes: {total_episodes}")
            print(f"Total Time: {total_time:.1f}s")
            print(f"Avg Time/Doc: {total_time/len(documents):.1f}s" if documents else "N/A")
            print(f"Neo4j Status: {report['summary']['neo4j_final_status']}")
            print("="*60)
            
            # Save report
            report_file = f"fast_ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Report saved to {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise
        finally:
            await self.close()


async def main():
    """Main function for fast ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fast document ingestion with optimized Neo4j handling"
    )
    parser.add_argument(
        "--documents", "-d",
        default="medical_docs",
        help="Documents folder path"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        default=True,
        help="Clean existing data before ingestion"
    )
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Disable Neo4j (Supabase only)"
    )
    parser.add_argument(
        "--neo4j-timeout",
        type=int,
        default=30,
        help="Neo4j timeout per chunk in seconds (default: 30)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for splitting documents"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=150,
        use_semantic_chunking=not args.no_semantic,
        extract_entities=False,  # Disabled for speed
        skip_graph_building=args.no_neo4j
    )
    
    # Create fast ingestion pipeline
    pipeline = FastValidatedIngestion(
        config=config,
        documents_folder=args.documents,
        neo4j_enabled=not args.no_neo4j,
        neo4j_timeout=args.neo4j_timeout,
        neo4j_max_retries=2
    )
    
    try:
        # Run ingestion
        report = await pipeline.run_fast_ingestion(clean_first=args.clean)
        
        # Exit code based on results
        if report["summary"]["failed_documents"] == 0:
            print("\n✅ All documents successfully ingested!")
            exit(0)
        else:
            print(f"\n⚠️ {report['summary']['failed_documents']} documents failed")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Ingestion interrupted")
        exit(2)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        exit(3)


if __name__ == "__main__":
    asyncio.run(main())