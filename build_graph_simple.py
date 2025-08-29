#!/usr/bin/env python3
"""
Simple knowledge graph builder that processes documents one at a time.
More robust with connection retries and smaller batch sizes.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from ingestion.graph_builder import GraphBuilder
from agent.unified_db_utils import initialize_database, close_database, execute_query
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def main():
    """Main execution function."""
    
    logger.info("=" * 60)
    logger.info("STARTING SIMPLE KNOWLEDGE GRAPH BUILDING")
    logger.info("=" * 60)
    
    # Initialize database connection
    await initialize_database()
    
    # Get documents from database
    logger.info("Fetching documents from database...")
    result = await execute_query(
        "SELECT id, title, source, content FROM documents ORDER BY created_at"
    )
    documents = result["data"] if result else []
    
    if not documents:
        logger.warning("No documents found in database")
        await close_database()
        return
    
    logger.info(f"Found {len(documents)} documents to process")
    
    # Initialize graph builder
    graph_builder = GraphBuilder()
    await graph_builder.initialize()
    
    total_episodes = 0
    total_errors = 0
    
    try:
        # Process each document
        for i, doc in enumerate(documents, 1):
            doc_id = doc["id"]
            doc_title = doc["title"]
            doc_source = doc["source"]
            
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing document {i}/{len(documents)}: {doc_title}")
            logger.info(f"{'='*40}")
            
            try:
                # Get chunks for this document
                chunk_result = await execute_query(
                    "SELECT * FROM chunks WHERE document_id = $1 ORDER BY chunk_index",
                    doc_id
                )
                chunks_data = chunk_result["data"] if chunk_result else []
                
                if not chunks_data:
                    logger.warning(f"No chunks found for document: {doc_title}")
                    continue
                
                logger.info(f"Found {len(chunks_data)} chunks")
                
                # Convert to DocumentChunk objects for graph builder
                from ingestion.chunker import DocumentChunk
                chunks = []
                for chunk_data in chunks_data:
                    chunk = DocumentChunk(
                        content=chunk_data["content"],
                        index=chunk_data["chunk_index"],
                        metadata=chunk_data.get("metadata", {}),
                        token_count=chunk_data.get("token_count", 0)
                    )
                    # Add embedding if available
                    if chunk_data.get("embedding"):
                        chunk.embedding = chunk_data["embedding"]
                    chunks.append(chunk)
                
                # Add to knowledge graph with small batch size
                logger.info("Building knowledge graph (this may take a few minutes)...")
                result = await graph_builder.add_document_to_graph(
                    chunks=chunks,
                    document_title=doc_title,
                    document_source=doc_source,
                    document_metadata={"document_id": doc_id},
                    batch_size=1,  # Process one chunk at a time for stability
                    group_id="0"  # Shared knowledge base
                )
                
                episodes = result.get("episodes_created", 0)
                errors = result.get("errors", [])
                
                total_episodes += episodes
                total_errors += len(errors)
                
                logger.info(f"✓ Created {episodes} episodes for: {doc_title}")
                
                if errors:
                    logger.warning(f"Encountered {len(errors)} errors:")
                    for error in errors[:3]:  # Show first 3 errors
                        logger.warning(f"  - {error}")
                
                # Delay between documents to avoid overwhelming Neo4j
                if i < len(documents):
                    logger.info("Waiting 3 seconds before next document...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                logger.error(f"Failed to process document {doc_title}: {e}")
                total_errors += 1
                continue
    
    finally:
        # Clean up
        await graph_builder.close()
        await close_database()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("KNOWLEDGE GRAPH BUILDING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Total episodes created: {total_episodes}")
    logger.info(f"Total errors: {total_errors}")
    
    if total_episodes > 0:
        logger.info("\n✅ Knowledge graph successfully populated!")
    else:
        logger.warning("\n⚠️ No episodes were created. Check the logs for errors.")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    print("\n⚠️  This will build the knowledge graph from existing documents.")
    print("Processing will be done one document at a time for stability.")
    print("This process may take 10-15 minutes.\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Knowledge graph building cancelled.")
        sys.exit(0)
    
    asyncio.run(main())