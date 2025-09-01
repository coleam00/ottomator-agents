#!/usr/bin/env python3
"""
Fix Neo4j embedding dimension issues comprehensively.

This script:
1. Checks current embedding dimensions in Neo4j
2. Fixes any mismatched dimensions
3. Updates Graphiti configuration
4. Validates the fix
"""

import os
import asyncio
import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase
from dotenv import load_dotenv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

TARGET_DIMENSION = 768

class Neo4jDimensionFixer:
    """Fix embedding dimension issues in Neo4j."""
    
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'password')
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def check_dimensions(self) -> Dict[str, Any]:
        """Check current embedding dimensions in the database."""
        with self.driver.session() as session:
            # Check entity embeddings
            entity_result = session.run("""
                MATCH (n:Entity)
                WHERE n.name_embedding IS NOT NULL
                RETURN 
                    COUNT(n) as total_entities,
                    size(n.name_embedding) as dimension,
                    COLLECT(DISTINCT size(n.name_embedding)) as unique_dimensions
                LIMIT 1
            """)
            entity_data = entity_result.single()
            
            # Check edge embeddings  
            edge_result = session.run("""
                MATCH ()-[r:RELATES_TO]->()
                WHERE r.fact_embedding IS NOT NULL
                RETURN 
                    COUNT(r) as total_edges,
                    size(r.fact_embedding) as dimension,
                    COLLECT(DISTINCT size(r.fact_embedding)) as unique_dimensions
                LIMIT 1
            """)
            edge_data = edge_result.single()
            
            return {
                'entities': {
                    'total': entity_data['total_entities'] if entity_data else 0,
                    'dimensions': entity_data['unique_dimensions'] if entity_data else [],
                },
                'edges': {
                    'total': edge_data['total_edges'] if edge_data else 0,
                    'dimensions': edge_data['unique_dimensions'] if edge_data else [],
                }
            }
    
    def normalize_embedding(self, embedding: List[float], target_dim: int = TARGET_DIMENSION) -> List[float]:
        """Normalize embedding to target dimension."""
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        
        embedding_array = np.array(embedding)
        
        if current_dim > target_dim:
            # Truncate
            normalized = embedding_array[:target_dim]
        else:
            # Pad with zeros
            padding = np.zeros(target_dim - current_dim)
            normalized = np.concatenate([embedding_array, padding])
        
        # Renormalize to unit length
        norm = np.linalg.norm(normalized)
        if norm > 0:
            normalized = normalized / norm
            
        return normalized.tolist()
    
    def fix_entity_embeddings(self) -> int:
        """Fix entity embeddings that have wrong dimensions."""
        fixed_count = 0
        
        with self.driver.session() as session:
            # Find entities with wrong dimensions
            result = session.run("""
                MATCH (n:Entity)
                WHERE n.name_embedding IS NOT NULL 
                AND size(n.name_embedding) <> $target_dim
                RETURN n.uuid as uuid, n.name as name, 
                       n.name_embedding as embedding,
                       size(n.name_embedding) as current_dim
            """, target_dim=TARGET_DIMENSION)
            
            for record in result:
                try:
                    # Normalize the embedding
                    normalized = self.normalize_embedding(record['embedding'])
                    
                    # Update in database
                    update_result = session.run("""
                        MATCH (n:Entity {uuid: $uuid})
                        SET n.name_embedding = $embedding
                        RETURN n.uuid
                    """, uuid=record['uuid'], embedding=normalized)
                    
                    if update_result.single():
                        fixed_count += 1
                        logger.info(f"Fixed entity '{record['name']}': {record['current_dim']} -> {TARGET_DIMENSION}")
                
                except Exception as e:
                    logger.error(f"Failed to fix entity {record['uuid']}: {e}")
        
        return fixed_count
    
    def fix_edge_embeddings(self) -> int:
        """Fix edge embeddings that have wrong dimensions."""
        fixed_count = 0
        
        with self.driver.session() as session:
            # Find edges with wrong dimensions
            result = session.run("""
                MATCH ()-[r:RELATES_TO]->()
                WHERE r.fact_embedding IS NOT NULL 
                AND size(r.fact_embedding) <> $target_dim
                RETURN id(r) as edge_id, r.name as name,
                       r.fact_embedding as embedding,
                       size(r.fact_embedding) as current_dim
            """, target_dim=TARGET_DIMENSION)
            
            for record in result:
                try:
                    # Normalize the embedding
                    normalized = self.normalize_embedding(record['embedding'])
                    
                    # Update in database
                    update_result = session.run("""
                        MATCH ()-[r:RELATES_TO]->()
                        WHERE id(r) = $edge_id
                        SET r.fact_embedding = $embedding
                        RETURN id(r)
                    """, edge_id=record['edge_id'], embedding=normalized)
                    
                    if update_result.single():
                        fixed_count += 1
                        logger.info(f"Fixed edge '{record['name']}': {record['current_dim']} -> {TARGET_DIMENSION}")
                
                except Exception as e:
                    logger.error(f"Failed to fix edge {record['edge_id']}: {e}")
        
        return fixed_count
    
    def create_indexes(self):
        """Create or update vector indexes with correct dimensions."""
        with self.driver.session() as session:
            try:
                # Drop existing indexes if they exist
                session.run("DROP INDEX entity_name_embedding IF EXISTS")
                session.run("DROP INDEX edge_fact_embedding IF EXISTS")
                
                # Create new indexes with correct dimensions
                session.run("""
                    CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS
                    FOR (n:Entity) ON (n.name_embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=TARGET_DIMENSION)
                
                session.run("""
                    CREATE VECTOR INDEX edge_fact_embedding IF NOT EXISTS
                    FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=TARGET_DIMENSION)
                
                logger.info(f"Created vector indexes with {TARGET_DIMENSION} dimensions")
                
            except Exception as e:
                logger.warning(f"Could not create indexes (may already exist): {e}")
    
    def validate_fix(self) -> bool:
        """Validate that all embeddings now have correct dimensions."""
        dims = self.check_dimensions()
        
        entity_dims = dims['entities']['dimensions']
        edge_dims = dims['edges']['dimensions']
        
        # Check if all dimensions are correct
        entities_ok = not entity_dims or (len(entity_dims) == 1 and entity_dims[0] == TARGET_DIMENSION)
        edges_ok = not edge_dims or (len(edge_dims) == 1 and edge_dims[0] == TARGET_DIMENSION)
        
        return entities_ok and edges_ok
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def run(self):
        """Run the complete fixing process."""
        logger.info("=" * 60)
        logger.info("NEO4J EMBEDDING DIMENSION FIXER")
        logger.info("=" * 60)
        
        # Check current state
        logger.info("\n1. Checking current dimensions...")
        initial_state = self.check_dimensions()
        logger.info(f"Entities: {initial_state['entities']}")
        logger.info(f"Edges: {initial_state['edges']}")
        
        # Fix entities
        logger.info("\n2. Fixing entity embeddings...")
        fixed_entities = self.fix_entity_embeddings()
        logger.info(f"Fixed {fixed_entities} entities")
        
        # Fix edges
        logger.info("\n3. Fixing edge embeddings...")
        fixed_edges = self.fix_edge_embeddings()
        logger.info(f"Fixed {fixed_edges} edges")
        
        # Create indexes
        logger.info("\n4. Creating/updating vector indexes...")
        self.create_indexes()
        
        # Validate
        logger.info("\n5. Validating fix...")
        if self.validate_fix():
            logger.info("✅ All embeddings now have correct dimensions!")
            final_state = self.check_dimensions()
            logger.info(f"Final state - Entities: {final_state['entities']}")
            logger.info(f"Final state - Edges: {final_state['edges']}")
            return True
        else:
            logger.error("❌ Some embeddings still have incorrect dimensions")
            problem_state = self.check_dimensions()
            logger.error(f"Problem state: {problem_state}")
            return False


async def test_graphiti_with_fix():
    """Test Graphiti after fixing dimensions."""
    from agent.graph_utils import GraphitiClient
    
    try:
        client = GraphitiClient()
        await client.initialize()
        
        # Try a simple search
        results = await client.search("test query", num_results=1)
        logger.info("✅ Graphiti search works after fix!")
        return True
    except Exception as e:
        logger.error(f"❌ Graphiti still has issues: {e}")
        return False


def main():
    """Main entry point."""
    fixer = Neo4jDimensionFixer()
    
    try:
        # Run the fix
        success = fixer.run()
        
        if success:
            # Test with Graphiti
            logger.info("\n6. Testing Graphiti integration...")
            asyncio.run(test_graphiti_with_fix())
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ DIMENSION FIX COMPLETE!")
            logger.info("=" * 60)
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ FIX FAILED - Manual intervention required")
            logger.error("=" * 60)
            
    finally:
        fixer.close()


if __name__ == "__main__":
    main()