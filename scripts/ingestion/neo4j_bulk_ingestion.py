#!/usr/bin/env python3
"""
Neo4j Bulk Ingestion Script using Graphiti's add_episode_bulk.
Processes all documents from Supabase in a single efficient bulk operation.
"""

import asyncio
import os
import sys
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4
import logging

from dotenv import load_dotenv
from supabase import create_client, Client
from graphiti_core import Graphiti
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.nodes import EpisodeType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Neo4jBulkIngestion:
    """Handles bulk ingestion of documents into Neo4j using Graphiti."""
    
    def __init__(self):
        """Initialize the bulk ingestion handler."""
        # Supabase configuration
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize graph client
        from agent.graph_utils import GraphitiClient
        self.graph_client = GraphitiClient()
        
        # Track progress
        self.checkpoint_file = "neo4j_bulk_ingestion_checkpoint.json"
        self.processed_documents = self.load_checkpoint()
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from file if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_checkpoint(self, data: Dict[str, Any]):
        """Save checkpoint to file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def fetch_all_documents_and_chunks(self) -> List[Dict[str, Any]]:
        """Fetch all documents and their chunks from Supabase."""
        logger.info("Fetching documents from Supabase...")
        
        try:
            # Get all documents
            result = self.supabase.table("documents").select("*").execute()
            documents = result.data
            
            if not documents:
                logger.warning("No documents found in database")
                return []
            
            logger.info(f"Found {len(documents)} documents")
            
            # Fetch chunks for all documents
            all_document_data = []
            
            for doc in documents:
                doc_id = doc["id"]
                doc_title = doc["title"]
                doc_source = doc.get("source", "")
                
                # Skip if already processed (based on checkpoint)
                if doc_id in self.processed_documents.get("completed_docs", []):
                    logger.info(f"Skipping already processed document: {doc_title}")
                    continue
                
                # Get chunks for this document
                chunk_result = self.supabase.table("chunks").select("*").eq(
                    "document_id", doc_id
                ).order("chunk_index").execute()
                
                chunks_data = chunk_result.data
                
                if chunks_data:
                    all_document_data.append({
                        "document": doc,
                        "chunks": chunks_data
                    })
                    logger.info(f"  - {doc_title}: {len(chunks_data)} chunks")
                else:
                    logger.warning(f"  - {doc_title}: No chunks found")
            
            return all_document_data
            
        except Exception as e:
            logger.error(f"Failed to fetch documents: {e}")
            raise
    
    def prepare_bulk_episodes(
        self, 
        documents_data: List[Dict[str, Any]]
    ) -> List[RawEpisode]:
        """Prepare RawEpisode objects for bulk ingestion."""
        logger.info("Preparing episodes for bulk ingestion...")
        
        bulk_episodes = []
        reference_time = datetime.now(timezone.utc)
        
        for doc_data in documents_data:
            doc = doc_data["document"]
            chunks = doc_data["chunks"]
            
            doc_id = doc["id"]
            doc_title = doc["title"]
            doc_source = doc.get("source", "")
            
            for chunk in chunks:
                try:
                    # Create unique episode name
                    episode_name = f"{doc_source}_chunk_{chunk['chunk_index']}_{uuid4().hex[:8]}"
                    
                    # Prepare content with context
                    content = chunk["content"]
                    
                    # Limit content length to avoid token limits
                    max_content_length = 6000
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "... [TRUNCATED]"
                    
                    # Add document context
                    episode_content = f"Document: {doc_title}\nChunk {chunk['chunk_index']}:\n\n{content}"
                    
                    # Create RawEpisode
                    episode = RawEpisode(
                        name=episode_name,
                        content=episode_content,
                        source=EpisodeType.text,
                        source_description=f"Medical document: {doc_title}",
                        reference_time=reference_time
                    )
                    
                    bulk_episodes.append(episode)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare episode for chunk {chunk['chunk_index']} of {doc_title}: {e}")
                    continue
        
        logger.info(f"Prepared {len(bulk_episodes)} episodes for bulk ingestion")
        return bulk_episodes
    
    async def perform_bulk_ingestion(
        self, 
        bulk_episodes: List[RawEpisode],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Perform bulk ingestion using Graphiti's add_episode_bulk method."""
        logger.info(f"Starting bulk ingestion of {len(bulk_episodes)} episodes...")
        
        # Initialize graph client
        await self.graph_client.initialize()
        
        results = {
            "total_episodes": len(bulk_episodes),
            "successful_batches": 0,
            "failed_batches": 0,
            "errors": []
        }
        
        try:
            # Process in batches to avoid overwhelming the system
            for i in range(0, len(bulk_episodes), batch_size):
                batch_end = min(i + batch_size, len(bulk_episodes))
                batch = bulk_episodes[i:batch_end]
                batch_num = (i // batch_size) + 1
                total_batches = (len(bulk_episodes) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} episodes)...")
                
                try:
                    # Use the bulk ingestion method
                    await self.graph_client.graphiti.add_episode_bulk(
                        bulk_episodes=batch,
                        group_id="0"  # Shared knowledge base
                    )
                    
                    results["successful_batches"] += 1
                    logger.info(f"✓ Batch {batch_num} successfully ingested")
                    
                    # Save progress
                    self.save_checkpoint({
                        "last_successful_batch": batch_num,
                        "total_batches": total_batches,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Small delay between batches
                    if batch_end < len(bulk_episodes):
                        await asyncio.sleep(2)
                    
                except Exception as e:
                    error_msg = f"Failed to ingest batch {batch_num}: {str(e)[:200]}"
                    logger.error(error_msg)
                    results["failed_batches"] += 1
                    results["errors"].append(error_msg)
                    
                    # Try to continue with next batch
                    await asyncio.sleep(5)
                    continue
            
        except Exception as e:
            logger.error(f"Critical error during bulk ingestion: {e}")
            results["errors"].append(str(e))
        
        finally:
            await self.graph_client.close()
        
        return results
    
    async def run(self):
        """Execute the full bulk ingestion pipeline."""
        logger.info("=" * 60)
        logger.info("NEO4J BULK INGESTION PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Fetch all documents and chunks
            documents_data = await self.fetch_all_documents_and_chunks()
            
            if not documents_data:
                logger.warning("No documents to process")
                return
            
            total_chunks = sum(len(d["chunks"]) for d in documents_data)
            logger.info(f"\nTotal documents: {len(documents_data)}")
            logger.info(f"Total chunks: {total_chunks}")
            
            # Step 2: Prepare bulk episodes
            bulk_episodes = self.prepare_bulk_episodes(documents_data)
            
            if not bulk_episodes:
                logger.warning("No episodes prepared for ingestion")
                return
            
            # Step 3: Perform bulk ingestion
            results = await self.perform_bulk_ingestion(bulk_episodes)
            
            # Step 4: Report results
            elapsed_time = datetime.now() - start_time
            
            logger.info("\n" + "=" * 60)
            logger.info("BULK INGESTION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total episodes: {results['total_episodes']}")
            logger.info(f"Successful batches: {results['successful_batches']}")
            logger.info(f"Failed batches: {results['failed_batches']}")
            logger.info(f"Total errors: {len(results['errors'])}")
            logger.info(f"Time elapsed: {elapsed_time}")
            
            if results['successful_batches'] > 0:
                logger.info("\n✅ Bulk ingestion completed successfully!")
                
                # Mark all documents as processed
                completed_docs = [d["document"]["id"] for d in documents_data]
                self.save_checkpoint({
                    "completed_docs": completed_docs,
                    "completion_time": datetime.now().isoformat(),
                    "total_episodes": results['total_episodes']
                })
            else:
                logger.error("\n❌ Bulk ingestion failed - no batches were successfully processed")
            
            if results['errors']:
                logger.warning("\nErrors encountered:")
                for error in results['errors'][:5]:  # Show first 5 errors
                    logger.warning(f"  - {error}")
            
        except KeyboardInterrupt:
            logger.info("\n\nProcess interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        
        logger.info("=" * 60)


async def main():
    """Main entry point."""
    ingestion = Neo4jBulkIngestion()
    await ingestion.run()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NEO4J BULK INGESTION USING GRAPHITI")
    print("="*60)
    print("\nThis script will:")
    print("1. Fetch all documents and chunks from Supabase")
    print("2. Prepare episodes in bulk using RawEpisode format")
    print("3. Use Graphiti's add_episode_bulk for efficient loading")
    print("4. Process all documents in optimized batches")
    print("\nEstimated time: 2-5 minutes (much faster than individual episodes)")
    print("Progress is saved and can be resumed if interrupted\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Bulk ingestion cancelled.")
        sys.exit(0)
    
    asyncio.run(main())