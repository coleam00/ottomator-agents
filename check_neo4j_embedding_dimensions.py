#!/usr/bin/env python3
"""
Check the actual embedding dimensions stored in Neo4j.
"""

import os
import asyncio
import logging
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_neo4j_embeddings():
    """Check the dimensions of embeddings stored in Neo4j."""
    
    # Neo4j configuration
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        logger.error("NEO4J_PASSWORD not set")
        return
    
    # Create Neo4j driver
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    try:
        async with driver.session() as session:
            # Query to check embedding dimensions in edges
            result = await session.run("""
                MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
                WHERE e.fact_embedding IS NOT NULL
                RETURN 
                    e.fact AS fact,
                    size(e.fact_embedding) AS embedding_dimension,
                    e.group_id AS group_id
                LIMIT 10
            """)
            
            records = await result.data()
            
            if records:
                logger.info(f"Found {len(records)} edges with embeddings:")
                dimensions = set()
                for record in records:
                    dim = record['embedding_dimension']
                    dimensions.add(dim)
                    logger.info(f"  - Fact: {record['fact'][:50]}... | Dimension: {dim} | Group: {record['group_id']}")
                
                if len(dimensions) > 1:
                    logger.error(f"❌ INCONSISTENT DIMENSIONS FOUND: {dimensions}")
                elif 3072 in dimensions:
                    logger.error(f"❌ EMBEDDINGS ARE 3072-DIMENSIONAL (should be 768)!")
                    logger.error("This is the root cause of the dimension mismatch error!")
                elif 768 in dimensions:
                    logger.info("✅ Embeddings are correctly 768-dimensional")
                else:
                    logger.warning(f"⚠️ Unexpected dimension: {dimensions}")
            else:
                logger.info("No edges with embeddings found in Neo4j")
            
            # Also check entity embeddings
            result = await session.run("""
                MATCH (n:Entity)
                WHERE n.entity_embedding IS NOT NULL
                RETURN 
                    n.name AS name,
                    size(n.entity_embedding) AS embedding_dimension
                LIMIT 10
            """)
            
            records = await result.data()
            
            if records:
                logger.info(f"\nFound {len(records)} entities with embeddings:")
                dimensions = set()
                for record in records:
                    dim = record['embedding_dimension']
                    dimensions.add(dim)
                    logger.info(f"  - Entity: {record['name']} | Dimension: {dim}")
                
                if 3072 in dimensions:
                    logger.error(f"❌ ENTITY EMBEDDINGS ARE 3072-DIMENSIONAL (should be 768)!")
                
    finally:
        await driver.close()


async def main():
    """Run the check."""
    logger.info("=" * 60)
    logger.info("Checking Neo4j Embedding Dimensions")
    logger.info("=" * 60)
    
    await check_neo4j_embeddings()
    
    logger.info("\n" + "=" * 60)
    logger.info("Check Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())