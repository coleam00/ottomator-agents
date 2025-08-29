#!/usr/bin/env python3
"""
Build knowledge graph from existing documents in the database.
This is a separate process that can be run after fast ingestion.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from agent.graph_utils import GraphitiClient
import logging
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class KnowledgeGraphBuilder:
    def __init__(self):
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize graph client
        self.graph_client = None
        
    async def initialize_graph_client(self):
        """Initialize the async graph client."""
        try:
            self.graph_client = GraphitiClient()
            await self.graph_client.initialize()
            logger.info("Graph client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize graph client: {e}")
            raise
    
    async def close_graph_client(self):
        """Close the graph client."""
        if self.graph_client:
            await self.graph_client.close()
    
    def get_documents(self) -> List[Dict]:
        """Get all documents from Supabase."""
        try:
            result = self.supabase.table("documents").select("*").execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            return []
    
    def get_chunks_for_document(self, document_id: str) -> List[Dict]:
        """Get all chunks for a specific document."""
        try:
            result = self.supabase.table("chunks").select("*").eq("document_id", document_id).order("chunk_index").execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching chunks for document {document_id}: {e}")
            return []
    
    async def build_graph_for_document(self, doc: Dict):
        """Build knowledge graph for a single document."""
        try:
            doc_title = doc.get('title', 'Unknown')
            doc_id = doc['id']
            
            logger.info(f"Building graph for: {doc_title}")
            
            # Get chunks for this document
            chunks = self.get_chunks_for_document(doc_id)
            
            if not chunks:
                logger.warning(f"No chunks found for document: {doc_title}")
                return
            
            # Prepare episode data
            episode_name = f"{doc_title}_{doc_id[:8]}"
            episode_body = "\n\n".join([chunk['content'] for chunk in chunks])
            
            logger.info(f"Creating episode with {len(chunks)} chunks")
            
            # Add episode to knowledge graph
            from datetime import datetime, timezone
            result = await self.graph_client.add_episode(
                episode_id=episode_name,
                content=episode_body,
                source=f"Medical document: {doc_title}",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "document_title": doc_title,
                    "document_id": doc_id,
                    "knowledge_type": "shared"
                },
                group_id="0"  # Shared knowledge base
            )
            
            if result:
                logger.info(f"✓ Successfully added episode for: {doc_title}")
            else:
                logger.warning(f"✗ Failed to add episode for: {doc_title}")
                
        except Exception as e:
            logger.error(f"Error building graph for document {doc.get('title', 'Unknown')}: {e}")
    
    async def build_incremental_graph(self):
        """Build knowledge graph incrementally for all documents."""
        logger.info("=" * 60)
        logger.info("STARTING INCREMENTAL KNOWLEDGE GRAPH BUILDING")
        logger.info("=" * 60)
        
        # Initialize graph client
        await self.initialize_graph_client()
        
        try:
            # Get all documents
            documents = self.get_documents()
            
            if not documents:
                logger.warning("No documents found in database")
                return
            
            logger.info(f"Found {len(documents)} documents to process")
            
            # Process each document incrementally
            for i, doc in enumerate(documents, 1):
                logger.info(f"\nProcessing document {i}/{len(documents)}")
                
                try:
                    await self.build_graph_for_document(doc)
                    
                    # Small delay between documents to avoid overwhelming Neo4j
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
                    logger.info("Continuing with next document...")
                    continue
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ Knowledge graph building complete!")
            logger.info("=" * 60)
            
        finally:
            # Close graph client
            await self.close_graph_client()

async def main():
    """Main execution function."""
    builder = KnowledgeGraphBuilder()
    
    try:
        await builder.build_incremental_graph()
        
        # Check Neo4j status after building
        logger.info("\nChecking Neo4j status...")
        os.system("python check_ingestion_status.py")
        
    except Exception as e:
        logger.error(f"Knowledge graph building failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("\n⚠️  This will build the knowledge graph from existing documents.")
    print("This process may take 5-10 minutes depending on the complexity.")
    print("The process can be interrupted and resumed if needed.\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Knowledge graph building cancelled.")
        sys.exit(0)
    
    asyncio.run(main())