#!/usr/bin/env python3
"""
Fast vector-only ingestion for Supabase.
Skips Neo4j/Graphiti for rapid document upload.
"""

import asyncio
import os
import sys
import glob
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.chunker import ChunkingConfig, create_chunker, DocumentChunk
from ingestion.embedder import create_embedder
from supabase import create_client, Client
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class FastVectorIngestion:
    """Fast ingestion pipeline for Supabase vector database only."""
    
    def __init__(self, documents_folder: str = "medical_docs", clean: bool = False):
        """Initialize fast ingestion pipeline."""
        self.documents_folder = documents_folder
        self.clean = clean
        self.supabase: Optional[Client] = None
        
        # Setup components
        self.chunker_config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=100,
            max_chunk_size=2000,
            use_semantic_splitting=False  # Disable for speed
        )
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": [],
            "start_time": None,
            "end_time": None
        }
    
    def initialize_supabase(self) -> Client:
        """Initialize Supabase client."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            raise ValueError("Missing Supabase credentials in environment")
        
        self.supabase = create_client(url, key)
        logger.info("Supabase client initialized")
        return self.supabase
    
    async def clean_database(self):
        """Clean existing data from Supabase."""
        if not self.clean:
            return
        
        logger.warning("Cleaning existing data from Supabase...")
        
        try:
            # Delete all chunks first (foreign key constraint)
            self.supabase.table("chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            logger.info("Deleted all chunks")
            
            # Delete all documents
            self.supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            logger.info("Deleted all documents")
            
        except Exception as e:
            logger.error(f"Error cleaning database: {e}")
            # Continue anyway
    
    def get_markdown_files(self) -> List[Path]:
        """Get all markdown files from documents folder."""
        pattern = os.path.join(self.documents_folder, "*.md")
        files = glob.glob(pattern)
        return sorted([Path(f) for f in files])
    
    async def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document."""
        result = {
            "file": file_path.name,
            "success": False,
            "chunks": 0,
            "error": None
        }
        
        try:
            # Read document
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title (first # heading or filename)
            lines = content.split('\n')
            title = file_path.stem.replace('_', ' ').title()
            for line in lines[:10]:
                if line.startswith('#'):
                    title = line.strip('#').strip()
                    break
            
            logger.info(f"Processing: {title}")
            
            # Create document record
            doc_data = {
                "title": title,
                "source": file_path.name,
                "content": content,
                "metadata": {
                    "ingestion_date": datetime.now(timezone.utc).isoformat(),
                    "file_size": len(content),
                    "ingestion_method": "fast_vector"
                }
            }
            
            # Insert document
            doc_response = self.supabase.table("documents").insert(doc_data).execute()
            if not doc_response.data:
                raise Exception("Failed to insert document")
            
            document_id = doc_response.data[0]["id"]
            logger.info(f"Created document: {document_id}")
            
            # Create chunks (synchronous)
            chunks = self.chunker.chunk_document(content, file_path.name, {"source": file_path.name})
            logger.info(f"Created {len(chunks)} chunks")
            
            # Generate embeddings in batches (async)
            # Extract text content from chunks for embedding
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedder.generate_embeddings_batch(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Prepare chunk records
            chunk_records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_record = {
                    "document_id": document_id,
                    "content": chunk.content,
                    "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    "chunk_index": i,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata or {}
                }
                chunk_records.append(chunk_record)
            
            # Insert chunks in batches of 50
            batch_size = 50
            for i in range(0, len(chunk_records), batch_size):
                batch = chunk_records[i:i+batch_size]
                self.supabase.table("chunks").insert(batch).execute()
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(chunk_records)-1)//batch_size + 1}")
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["chunks_created"] += len(chunks)
            self.stats["embeddings_generated"] += len(embeddings)
            
            result["success"] = True
            result["chunks"] = len(chunks)
            
            logger.info(f"✅ Successfully processed: {title} ({len(chunks)} chunks)")
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            logger.error(error_msg)
            result["error"] = str(e)
            self.stats["errors"].append(error_msg)
        
        return result
    
    async def run(self) -> Dict[str, Any]:
        """Run the fast ingestion pipeline."""
        self.stats["start_time"] = time.time()
        
        try:
            # Initialize Supabase
            self.initialize_supabase()
            
            # Clean database if requested
            if self.clean:
                await self.clean_database()
            
            # Get markdown files
            files = self.get_markdown_files()
            logger.info(f"Found {len(files)} markdown files to process")
            
            if not files:
                logger.warning("No markdown files found")
                return self.stats
            
            # Process each document
            results = []
            for i, file_path in enumerate(files, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing document {i}/{len(files)}")
                logger.info(f"{'='*60}")
                
                result = await self.process_document(file_path)
                results.append(result)
                
                # Brief pause to avoid rate limits
                if i < len(files):
                    await asyncio.sleep(0.5)
            
            # Summary
            self.stats["end_time"] = time.time()
            duration = self.stats["end_time"] - self.stats["start_time"]
            
            logger.info("\n" + "="*60)
            logger.info("INGESTION COMPLETE")
            logger.info("="*60)
            logger.info(f"Documents processed: {self.stats['documents_processed']}/{len(files)}")
            logger.info(f"Total chunks created: {self.stats['chunks_created']}")
            logger.info(f"Total embeddings generated: {self.stats['embeddings_generated']}")
            logger.info(f"Errors: {len(self.stats['errors'])}")
            logger.info(f"Time taken: {duration:.2f} seconds")
            logger.info(f"Average time per document: {duration/len(files):.2f} seconds")
            
            if self.stats["errors"]:
                logger.error("\nErrors encountered:")
                for error in self.stats["errors"]:
                    logger.error(f"  - {error}")
            
            # Save results to file
            results_file = f"ingestion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "stats": self.stats,
                    "results": results,
                    "duration_seconds": duration
                }, f, indent=2)
            logger.info(f"\nResults saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Fatal error in ingestion pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.stats["errors"].append(f"Fatal: {str(e)}")
        
        return self.stats


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast vector ingestion for Supabase")
    parser.add_argument("--clean", action="store_true", help="Clean existing data before ingestion")
    parser.add_argument("--folder", default="medical_docs", help="Folder containing markdown documents")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 FAST VECTOR INGESTION (Supabase Only)")
    print("="*60)
    print(f"Documents folder: {args.folder}")
    print(f"Clean before ingest: {args.clean}")
    print("="*60 + "\n")
    
    if args.clean:
        response = input("⚠️  WARNING: This will delete all existing data. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    ingestion = FastVectorIngestion(
        documents_folder=args.folder,
        clean=args.clean
    )
    
    stats = await ingestion.run()
    
    # Exit with error code if any documents failed
    if stats["errors"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())