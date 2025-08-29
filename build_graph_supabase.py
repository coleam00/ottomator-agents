#!/usr/bin/env python3
"""
Knowledge graph builder specifically for Supabase backend.
Processes documents one at a time with proper error handling.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client
from ingestion.graph_builder import GraphBuilder
from ingestion.chunker import DocumentChunk
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
    logger.info("KNOWLEDGE GRAPH BUILDER FOR SUPABASE")
    logger.info("=" * 60)
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase credentials not found in environment variables")
        sys.exit(1)
    
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("Connected to Supabase")
    
    # Get documents from Supabase
    logger.info("Fetching documents from database...")
    try:
        result = supabase.table("documents").select("*").execute()
        documents = result.data
    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}")
        sys.exit(1)
    
    if not documents:
        logger.warning("No documents found in database")
        return
    
    logger.info(f"Found {len(documents)} documents to process")
    
    # Initialize graph builder
    graph_builder = GraphBuilder()
    await graph_builder.initialize()
    
    total_episodes = 0
    total_errors = 0
    successful_docs = 0
    
    try:
        # Process each document
        for i, doc in enumerate(documents, 1):
            doc_id = doc["id"]
            doc_title = doc["title"]
            doc_source = doc.get("source", "")
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Document {i}/{len(documents)}: {doc_title}")
            logger.info(f"{'='*50}")
            
            try:
                # Get chunks for this document from Supabase
                chunk_result = supabase.table("chunks").select("*").eq("document_id", doc_id).order("chunk_index").execute()
                chunks_data = chunk_result.data
                
                if not chunks_data:
                    logger.warning(f"No chunks found for document: {doc_title}")
                    continue
                
                logger.info(f"Found {len(chunks_data)} chunks for this document")
                
                # Convert to DocumentChunk objects
                chunks = []
                for chunk_data in chunks_data:
                    content = chunk_data["content"]
                    chunk = DocumentChunk(
                        content=content,
                        index=chunk_data["chunk_index"],
                        start_char=0,  # Not stored in DB, use defaults
                        end_char=len(content),  # Use content length as approximation
                        metadata=chunk_data.get("metadata", {}),
                        token_count=chunk_data.get("token_count", 0)
                    )
                    # Add embedding if available
                    if chunk_data.get("embedding"):
                        chunk.embedding = chunk_data["embedding"]
                    chunks.append(chunk)
                
                # Process chunks in smaller batches to avoid timeouts
                # Split chunks into batches of 3
                batch_size = 3
                for batch_start in range(0, len(chunks), batch_size):
                    batch_end = min(batch_start + batch_size, len(chunks))
                    batch_chunks = chunks[batch_start:batch_end]
                    
                    logger.info(f"Processing chunks {batch_start+1}-{batch_end} of {len(chunks)}...")
                    
                    # Add batch to knowledge graph
                    result = await graph_builder.add_document_to_graph(
                        chunks=batch_chunks,
                        document_title=f"{doc_title} (part {batch_start//batch_size + 1})",
                        document_source=doc_source,
                        document_metadata={
                            "document_id": doc_id,
                            "batch": f"{batch_start+1}-{batch_end}"
                        },
                        batch_size=1,  # Process one chunk at a time within the batch
                        group_id="0"  # Shared knowledge base
                    )
                    
                    episodes = result.get("episodes_created", 0)
                    errors = result.get("errors", [])
                    
                    total_episodes += episodes
                    total_errors += len(errors)
                    
                    if episodes > 0:
                        logger.info(f"✓ Created {episodes} episodes for batch {batch_start+1}-{batch_end}")
                    
                    if errors:
                        logger.warning(f"Errors in batch {batch_start+1}-{batch_end}:")
                        for error in errors[:2]:  # Show first 2 errors
                            logger.warning(f"  - {error[:100]}...")  # Truncate long errors
                    
                    # Small delay between batches
                    if batch_end < len(chunks):
                        await asyncio.sleep(2)
                
                successful_docs += 1
                logger.info(f"✅ Completed document: {doc_title}")
                
                # Delay between documents
                if i < len(documents):
                    logger.info("Waiting 3 seconds before next document...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                logger.error(f"Failed to process document {doc_title}: {str(e)[:200]}")
                total_errors += 1
                
                # Try to continue with next document
                logger.info("Continuing with next document...")
                await asyncio.sleep(5)  # Longer delay after error
                continue
    
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
    
    finally:
        # Clean up
        logger.info("\nCleaning up connections...")
        await graph_builder.close()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("KNOWLEDGE GRAPH BUILDING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Documents processed: {successful_docs}/{len(documents)}")
    logger.info(f"Total episodes created: {total_episodes}")
    logger.info(f"Total errors encountered: {total_errors}")
    
    if total_episodes > 0:
        logger.info("\n✅ Knowledge graph successfully populated!")
        logger.info(f"Average episodes per document: {total_episodes/successful_docs:.1f}")
    else:
        logger.warning("\n⚠️ No episodes were created. Check the logs for errors.")
    
    logger.info("=" * 60)
    
    # Check Neo4j status
    logger.info("\nChecking Neo4j database status...")
    os.system("python check_neo4j_status.py")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH BUILDER FOR MEDICAL DOCUMENTS")
    print("="*60)
    print("\nThis script will:")
    print("1. Connect to your Supabase database")
    print("2. Fetch all documents and their chunks")
    print("3. Build a knowledge graph in Neo4j using Graphiti")
    print("4. Process documents in small batches to avoid timeouts")
    print("\nEstimated time: 10-20 minutes for 11 documents")
    print("The process can be interrupted with Ctrl+C and resumed later\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Knowledge graph building cancelled.")
        sys.exit(0)
    
    asyncio.run(main())