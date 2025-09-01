#!/usr/bin/env python3
"""
Complete Neo4j Ingestion for Remaining Documents
Using a simplified direct approach to avoid timeout issues.
"""

import asyncio
import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SimpleNeo4jIngestion:
    """Simple and direct Neo4j ingestion without complex Graphiti processing."""
    
    def __init__(self):
        """Initialize ingestion."""
        # Neo4j configuration
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_password]):
            raise ValueError("Neo4j credentials not configured")
        
        self.driver = None
        self.processed_docs = set()
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize Neo4j connection."""
        logger.info("Connecting to Neo4j...")
        
        config = {
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": 50,
            "connection_acquisition_timeout": 30,
            "connection_timeout": 10,
            "keep_alive": True,
        }
        
        self.driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
            **config
        )
        
        # Test connection
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            logger.info("✅ Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def check_existing_documents(self):
        """Check which documents are already in Neo4j."""
        logger.info("Checking existing documents in Neo4j...")
        
        try:
            async with self.driver.session() as session:
                # Get existing episodes or chunks
                result = await session.run("""
                    MATCH (n)
                    WHERE n:Episode OR n:Chunk
                    RETURN DISTINCT n.source as source, count(*) as count
                """)
                
                records = await result.data()
                for record in records:
                    if record['source']:
                        # Extract document name from source
                        source = record['source']
                        logger.info(f"  Found: {source} ({record['count']} nodes)")
                        # Add to processed set
                        if 'doc' in source.lower():
                            self.processed_docs.add(source)
                
                # Also check for document nodes if they exist
                result = await session.run("""
                    MATCH (d:Document)
                    RETURN d.id as id, d.title as title
                """)
                
                records = await result.data()
                for record in records:
                    doc_id = record['id']
                    logger.info(f"  Found document: {doc_id}")
                    self.processed_docs.add(doc_id)
                
        except Exception as e:
            logger.warning(f"Error checking existing documents: {e}")
        
        logger.info(f"Found {len(self.processed_docs)} documents already processed")
        return self.processed_docs
    
    async def load_documents(self, docs_dir: str = "medical_docs") -> List[Dict[str, Any]]:
        """Load documents from directory."""
        logger.info(f"Loading documents from {docs_dir}...")
        
        documents = []
        doc_path = Path(docs_dir)
        
        if not doc_path.exists():
            logger.error(f"Directory {docs_dir} not found")
            return documents
        
        # Get all markdown files
        md_files = sorted(doc_path.glob("*.md"))
        
        for file_path in md_files:
            doc_id = file_path.stem
            
            # Skip if already processed
            if doc_id in self.processed_docs or file_path.name in self.processed_docs:
                logger.info(f"  ⏭️ Skipping already processed: {file_path.name}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create simple chunks
                chunk_size = 1000
                chunks = []
                for i in range(0, len(content), chunk_size):
                    chunk_text = content[i:i+chunk_size]
                    chunks.append({
                        "index": len(chunks),
                        "text": chunk_text,
                        "start": i,
                        "end": min(i + chunk_size, len(content))
                    })
                
                doc = {
                    "id": doc_id,
                    "title": doc_id.replace('_', ' ').title(),
                    "source": file_path.name,
                    "content": content,
                    "chunks": chunks,
                    "file_path": str(file_path)
                }
                
                documents.append(doc)
                logger.info(f"  ✓ Loaded: {file_path.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} new documents to process")
        return documents
    
    async def ingest_document(self, doc: Dict[str, Any]) -> bool:
        """Ingest a single document into Neo4j."""
        doc_id = doc['id']
        doc_title = doc['title']
        
        logger.info(f"Processing: {doc_title}...")
        
        try:
            async with self.driver.session() as session:
                # Create document node
                await session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.title = $title,
                        d.source = $source,
                        d.created_at = timestamp(),
                        d.group_id = '0'
                    RETURN d
                """, id=doc_id, title=doc_title, source=doc['source'])
                
                # Create chunks and basic entities
                for chunk in doc['chunks']:
                    chunk_id = f"{doc_id}_chunk_{chunk['index']}"
                    
                    # Create chunk node
                    await session.run("""
                        MERGE (c:Chunk {id: $chunk_id})
                        SET c.content = $content,
                            c.index = $index,
                            c.document_id = $doc_id,
                            c.source = $source,
                            c.group_id = '0',
                            c.created_at = timestamp()
                        WITH c
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[:HAS_CHUNK]->(c)
                        RETURN c
                    """, 
                    chunk_id=chunk_id,
                    content=chunk['text'][:2000],  # Limit content size
                    index=chunk['index'],
                    doc_id=doc_id,
                    source=doc['source']
                    )
                    
                    self.stats["chunks_created"] += 1
                    
                    # Extract simple entities from chunk (basic keywords)
                    entities = self.extract_simple_entities(chunk['text'])
                    
                    for entity in entities:
                        # Create entity node
                        await session.run("""
                            MERGE (e:Entity {name: $name})
                            SET e.type = 'medical_concept',
                                e.group_id = '0'
                            WITH e
                            MATCH (c:Chunk {id: $chunk_id})
                            MERGE (c)-[:MENTIONS]->(e)
                            RETURN e
                        """, name=entity, chunk_id=chunk_id)
                        
                        self.stats["entities_created"] += 1
                
                # Create episode node for the document
                episode_id = f"episode_{doc_id}_{uuid4().hex[:8]}"
                await session.run("""
                    CREATE (e:Episode {
                        id: $episode_id,
                        name: $name,
                        content: $content,
                        source: $source,
                        document_id: $doc_id,
                        group_id: '0',
                        created_at: timestamp()
                    })
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_EPISODE]->(e)
                    RETURN e
                """,
                episode_id=episode_id,
                name=f"Medical Document: {doc_title}",
                content=doc['content'][:5000],  # Store truncated content
                source=doc['source'],
                doc_id=doc_id
                )
                
                self.stats["documents_processed"] += 1
                logger.info(f"  ✅ Successfully ingested {doc_title}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to ingest {doc_title}: {e}")
            self.stats["errors"] += 1
            return False
    
    def extract_simple_entities(self, text: str) -> List[str]:
        """Extract simple medical entities from text."""
        # Basic keyword extraction for medical terms
        keywords = [
            "menopause", "perimenopause", "estrogen", "progesterone", "hormone",
            "hot flash", "night sweat", "vaginal dryness", "mood swing", "anxiety",
            "depression", "bone loss", "osteoporosis", "weight gain", "metabolism",
            "HRT", "hormone therapy", "supplement", "vitamin", "calcium",
            "vitamin D", "black cohosh", "soy", "isoflavone", "DHEA",
            "mindfulness", "exercise", "diet", "sleep", "stress"
        ]
        
        entities = []
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                entities.append(keyword.title())
        
        # Return unique entities
        return list(set(entities))[:10]  # Limit to 10 per chunk
    
    async def create_indices(self):
        """Create Neo4j indices for better performance."""
        logger.info("Creating indices...")
        
        indices = [
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (ep:Episode) ON (ep.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.group_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Chunk) ON (n.group_id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.group_id)"
        ]
        
        async with self.driver.session() as session:
            for index_query in indices:
                try:
                    await session.run(index_query)
                except Exception as e:
                    logger.debug(f"Index creation: {e}")
    
    async def run(self):
        """Execute the ingestion pipeline."""
        logger.info("="*60)
        logger.info("NEO4J DOCUMENT INGESTION")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Initialize connection
            await self.initialize()
            
            # Create indices
            await self.create_indices()
            
            # Check existing documents
            await self.check_existing_documents()
            
            # Load new documents
            documents = await self.load_documents()
            
            if not documents:
                logger.info("\n✅ All documents are already ingested!")
                return
            
            # Process each document
            logger.info(f"\nIngesting {len(documents)} documents...")
            for i, doc in enumerate(documents, 1):
                logger.info(f"\n[{i}/{len(documents)}] Processing {doc['title']}")
                success = await self.ingest_document(doc)
                
                if success:
                    # Small delay between documents
                    await asyncio.sleep(1)
            
            # Report statistics
            elapsed = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info("INGESTION COMPLETE")
            logger.info("="*60)
            logger.info(f"Time elapsed: {elapsed:.2f} seconds")
            logger.info(f"Documents processed: {self.stats['documents_processed']}")
            logger.info(f"Chunks created: {self.stats['chunks_created']}")
            logger.info(f"Entities created: {self.stats['entities_created']}")
            logger.info(f"Errors: {self.stats['errors']}")
            
            if self.stats['errors'] == 0:
                logger.info("\n✅ All documents successfully ingested!")
            else:
                logger.warning(f"\n⚠️ Completed with {self.stats['errors']} errors")
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        
        finally:
            # Close connection
            if self.driver:
                await self.driver.close()
                logger.info("\nConnection closed")


async def main():
    """Main entry point."""
    ingestion = SimpleNeo4jIngestion()
    await ingestion.run()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIMPLE NEO4J INGESTION")
    print("="*60)
    print("\nThis script will:")
    print("  1. Check for already ingested documents")
    print("  2. Load remaining medical documents")
    print("  3. Create simple graph structure in Neo4j")
    print("  4. Extract basic medical entities")
    print("\nEstimated time: 1-2 minutes\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response != "yes":
        print("Ingestion cancelled.")
        sys.exit(0)
    
    asyncio.run(main())