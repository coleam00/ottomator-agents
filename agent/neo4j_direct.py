"""
Direct Neo4j operations for knowledge base queries.

This module provides direct Neo4j access for searching the medical knowledge base
that was ingested directly into Neo4j, bypassing Graphiti. It handles:
- Direct Cypher queries for graph traversal
- Entity and relationship searches
- Knowledge base fact retrieval
- Graph structure exploration
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Neo4jDirectClient:
    """Direct Neo4j client for knowledge base queries."""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        Initialize direct Neo4j client.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")
        
        self.driver = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Neo4j driver."""
        if self._initialized:
            return
        
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Test connection
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            self._initialized = True
            logger.info("Direct Neo4j client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise
    
    async def close(self):
        """Close Neo4j driver."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            self._initialized = False
            logger.info("Direct Neo4j client closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get an async Neo4j session."""
        if not self._initialized:
            await self.initialize()
        
        session = self.driver.session()
        try:
            yield session
        finally:
            await session.close()
    
    async def search_knowledge_base(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base using direct Neo4j queries.
        
        This searches for entities and relationships that match the query.
        Optimized for the medical knowledge base that was ingested directly.
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching facts and relationships
        """
        try:
            async with self.get_session() as session:
                # Search for entities matching the query
                entity_query = """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($query)
                   OR (n.description IS NOT NULL AND toLower(n.description) CONTAINS toLower($query))
                OPTIONAL MATCH (n)-[r]-(related:Entity)
                RETURN DISTINCT 
                    n.name as entity,
                    n.description as description,
                    collect(DISTINCT {
                        type: type(r),
                        related_entity: related.name,
                        related_description: related.description
                    }) as relationships
                LIMIT $limit
                """
                
                result = await session.run(entity_query, parameters={"query": query, "limit": limit})
                entities = []
                async for record in result:
                    entities.append({
                        "type": "entity",
                        "entity": record["entity"],
                        "description": record["description"],
                        "relationships": [
                            rel for rel in record["relationships"] 
                            if rel["related_entity"] is not None
                        ]
                    })
                
                # Search for relationships/facts containing the query
                fact_query = """
                MATCH (e1:Entity)-[r]-(e2:Entity)
                WHERE toLower(e1.name) CONTAINS toLower($query)
                   OR toLower(e2.name) CONTAINS toLower($query)
                   OR (r.description IS NOT NULL AND toLower(r.description) CONTAINS toLower($query))
                   OR (type(r) IS NOT NULL AND toLower(type(r)) CONTAINS toLower($query))
                RETURN DISTINCT
                    e1.name as subject,
                    type(r) as relationship,
                    e2.name as object,
                    r.description as description,
                    e1.name + ' ' + type(r) + ' ' + e2.name as fact
                LIMIT $limit
                """
                
                result = await session.run(fact_query, parameters={"query": query, "limit": limit})
                facts = []
                async for record in result:
                    facts.append({
                        "type": "fact",
                        "fact": record["fact"],
                        "subject": record["subject"],
                        "relationship": record["relationship"],
                        "object": record["object"],
                        "description": record["description"]
                    })
                
                # Combine and return results
                results = entities + facts
                
                # Sort by relevance (simple scoring based on query match)
                for result in results:
                    score = 0
                    query_lower = query.lower()
                    
                    # Score based on matches
                    if result.get("entity") and query_lower in result["entity"].lower():
                        score += 10
                    if result.get("fact") and query_lower in result["fact"].lower():
                        score += 8
                    if result.get("description") and result["description"] and query_lower in result["description"].lower():
                        score += 5
                    
                    result["relevance_score"] = score
                
                # Sort by relevance
                results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                
                # Limit final results
                return results[:limit]
                
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []
    
    async def get_entity_relationships(
        self,
        entity: str,
        depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get all relationships for a specific entity up to a certain depth.
        
        Args:
            entity: Entity name to search for
            depth: Maximum traversal depth (1-3 recommended)
            relationship_types: Optional list of relationship types to filter
        
        Returns:
            Dictionary containing the entity and its relationships
        """
        try:
            async with self.get_session() as session:
                # Build relationship filter if provided
                rel_filter = ""
                if relationship_types:
                    rel_types_str = "|".join(relationship_types)
                    rel_filter = f"[r:{rel_types_str}*1..{depth}]"
                else:
                    rel_filter = f"[r*1..{depth}]"
                
                # Query for entity and its relationships
                query = f"""
                MATCH path = (e:Entity {{name: $entity}})-{rel_filter}-(related:Entity)
                WITH e, path, related, relationships(path) as rels
                RETURN DISTINCT
                    e.name as central_entity,
                    e.description as central_description,
                    collect(DISTINCT {{
                        related_entity: related.name,
                        related_description: related.description,
                        path_length: length(path),
                        relationships: [r in rels | {{
                            type: type(r),
                            description: r.description
                        }}]
                    }}) as connections
                """
                
                result = await session.run(query, parameters={"entity": entity})
                record = await result.single()
                
                if not record:
                    # Try case-insensitive search
                    case_insensitive_query = f"""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = toLower($entity)
                    WITH e LIMIT 1
                    MATCH path = (e)-{rel_filter}-(related:Entity)
                    WITH e, path, related, relationships(path) as rels
                    RETURN DISTINCT
                        e.name as central_entity,
                        e.description as central_description,
                        collect(DISTINCT {{
                            related_entity: related.name,
                            related_description: related.description,
                            path_length: length(path),
                            relationships: [r in rels | {{
                                type: type(r),
                                description: r.description
                            }}]
                        }}) as connections
                    """
                    
                    result = await session.run(case_insensitive_query, parameters={"entity": entity})
                    record = await result.single()
                
                if record:
                    return {
                        "central_entity": record["central_entity"],
                        "central_description": record["central_description"],
                        "connections": record["connections"],
                        "depth": depth,
                        "total_connections": len(record["connections"])
                    }
                else:
                    return {
                        "central_entity": entity,
                        "central_description": None,
                        "connections": [],
                        "depth": depth,
                        "total_connections": 0,
                        "error": f"Entity '{entity}' not found in knowledge base"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get entity relationships: {e}")
            return {
                "central_entity": entity,
                "connections": [],
                "depth": depth,
                "error": str(e)
            }
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            async with self.get_session() as session:
                stats = {}
                
                # Count entities
                result = await session.run("MATCH (n:Entity) RETURN count(n) as count")
                record = await result.single()
                stats["total_entities"] = record["count"]
                
                # Count relationships
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = await result.single()
                stats["total_relationships"] = record["count"]
                
                # Get node labels
                result = await session.run("CALL db.labels()")
                labels = []
                async for record in result:
                    labels.append(record["label"])
                stats["node_labels"] = labels
                
                # Get relationship types
                result = await session.run("CALL db.relationshipTypes()")
                rel_types = []
                async for record in result:
                    rel_types.append(record["relationshipType"])
                stats["relationship_types"] = rel_types
                
                # Count by label
                label_counts = {}
                for label in labels:
                    result = await session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    record = await result.single()
                    label_counts[label] = record["count"]
                stats["nodes_by_label"] = label_counts
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}
    
    async def find_paths_between_entities(
        self,
        entity1: str,
        entity2: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find all paths between two entities.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            max_depth: Maximum path length
        
        Returns:
            List of paths between the entities
        """
        try:
            async with self.get_session() as session:
                query = """
                MATCH path = shortestPath(
                    (e1:Entity {name: $entity1})-[*..%d]-(e2:Entity {name: $entity2})
                )
                RETURN 
                    [n in nodes(path) | n.name] as entities,
                    [r in relationships(path) | type(r)] as relationships,
                    length(path) as path_length
                """ % max_depth
                
                result = await session.run(query, parameters={"entity1": entity1, "entity2": entity2})
                paths = []
                
                async for record in result:
                    path_info = {
                        "entities": record["entities"],
                        "relationships": record["relationships"],
                        "path_length": record["path_length"],
                        "path_description": self._format_path(
                            record["entities"], 
                            record["relationships"]
                        )
                    }
                    paths.append(path_info)
                
                if not paths:
                    # Try case-insensitive search
                    case_insensitive_query = """
                    MATCH (e1:Entity), (e2:Entity)
                    WHERE toLower(e1.name) = toLower($entity1)
                      AND toLower(e2.name) = toLower($entity2)
                    WITH e1, e2 LIMIT 1
                    MATCH path = shortestPath((e1)-[*..%d]-(e2))
                    RETURN 
                        [n in nodes(path) | n.name] as entities,
                        [r in relationships(path) | type(r)] as relationships,
                        length(path) as path_length
                    """ % max_depth
                    
                    result = await session.run(
                        case_insensitive_query, 
                        parameters={"entity1": entity1, "entity2": entity2}
                    )
                    
                    async for record in result:
                        path_info = {
                            "entities": record["entities"],
                            "relationships": record["relationships"],
                            "path_length": record["path_length"],
                            "path_description": self._format_path(
                                record["entities"], 
                                record["relationships"]
                            )
                        }
                        paths.append(path_info)
                
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find paths between entities: {e}")
            return []
    
    def _format_path(self, entities: List[str], relationships: List[str]) -> str:
        """
        Format a path as a readable string.
        
        Args:
            entities: List of entity names in the path
            relationships: List of relationship types in the path
        
        Returns:
            Formatted path string
        """
        if not entities:
            return ""
        
        path_parts = [entities[0]]
        for i, rel in enumerate(relationships):
            if i + 1 < len(entities):
                path_parts.append(f"-[{rel}]->")
                path_parts.append(entities[i + 1])
        
        return " ".join(path_parts)
    
    async def get_entity_by_type(
        self,
        entity_type: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type or with specific properties.
        
        Args:
            entity_type: Type or property to filter entities
            limit: Maximum number of results
        
        Returns:
            List of entities matching the type
        """
        try:
            async with self.get_session() as session:
                # First try to find entities with a specific label
                query = """
                MATCH (n)
                WHERE $entity_type IN labels(n)
                RETURN 
                    n.name as name,
                    n.description as description,
                    labels(n) as labels
                LIMIT $limit
                """
                
                result = await session.run(query, parameters={"entity_type": entity_type, "limit": limit})
                entities = []
                
                async for record in result:
                    entities.append({
                        "name": record["name"],
                        "description": record["description"],
                        "labels": record["labels"]
                    })
                
                # If no results, try searching in properties
                if not entities:
                    property_query = """
                    MATCH (n:Entity)
                    WHERE toLower(n.type) = toLower($entity_type)
                       OR toLower(n.category) = toLower($entity_type)
                       OR any(label IN labels(n) WHERE toLower(label) CONTAINS toLower($entity_type))
                    RETURN 
                        n.name as name,
                        n.description as description,
                        labels(n) as labels,
                        n.type as type,
                        n.category as category
                    LIMIT $limit
                    """
                    
                    result = await session.run(
                        property_query, 
                        parameters={"entity_type": entity_type, "limit": limit}
                    )
                    
                    async for record in result:
                        entities.append({
                            "name": record["name"],
                            "description": record["description"],
                            "labels": record["labels"],
                            "type": record.get("type"),
                            "category": record.get("category")
                        })
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to get entities by type: {e}")
            return []


# Global Neo4j direct client instance
neo4j_direct_client = Neo4jDirectClient()


async def initialize_neo4j_direct():
    """Initialize direct Neo4j client."""
    await neo4j_direct_client.initialize()


async def close_neo4j_direct():
    """Close direct Neo4j client."""
    await neo4j_direct_client.close()


# Convenience functions for common operations
async def search_knowledge_base_direct(
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search the knowledge base using direct Neo4j queries.
    
    Args:
        query: Search query
        limit: Maximum number of results
    
    Returns:
        Search results from knowledge base
    """
    return await neo4j_direct_client.search_knowledge_base(query, limit)


async def get_entity_relationships_direct(
    entity: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get relationships for an entity using direct Neo4j queries.
    
    Args:
        entity: Entity name
        depth: Maximum traversal depth
    
    Returns:
        Entity relationships
    """
    return await neo4j_direct_client.get_entity_relationships(entity, depth)


async def find_entity_paths(
    entity1: str,
    entity2: str,
    max_depth: int = 3
) -> List[Dict[str, Any]]:
    """
    Find paths between two entities.
    
    Args:
        entity1: First entity
        entity2: Second entity
        max_depth: Maximum path length
    
    Returns:
        List of paths between entities
    """
    return await neo4j_direct_client.find_paths_between_entities(
        entity1, entity2, max_depth
    )


async def test_neo4j_direct_connection() -> bool:
    """
    Test direct Neo4j connection.
    
    Returns:
        True if connection successful
    """
    try:
        await neo4j_direct_client.initialize()
        stats = await neo4j_direct_client.get_graph_statistics()
        logger.info(f"Direct Neo4j connection successful. Stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Direct Neo4j connection test failed: {e}")
        return False