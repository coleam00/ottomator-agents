#!/usr/bin/env python
"""Check detailed Neo4j content."""

from neo4j import GraphDatabase
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    print("Error: Missing Neo4j environment variables")
    print("Required: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
    sys.exit(1)

try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        connection_timeout=10.0,  # 10 second timeout
        max_connection_lifetime=3600  # 1 hour
    )
except Exception as e:
    print(f"Failed to create Neo4j driver: {e}")
    sys.exit(1)

try:
    with driver.session() as session:
        # Check for Episode nodes
        result = session.run("MATCH (e:Episode) RETURN e.episode_id as id, e.valid_at as timestamp ORDER BY e.valid_at DESC LIMIT 10")
        episodes = list(result)
        
        print(f"\nFound {len(episodes)} Episode nodes:")
        for ep in episodes:
            print(f"  - {ep['id']}")
        
        # Check for unique entity types
        result = session.run("MATCH (n:Entity) RETURN DISTINCT n.entity_types as types, count(*) as count ORDER BY count DESC LIMIT 10")
        entity_types = list(result)
        
        print(f"\nEntity type distribution:")
        for et in entity_types:
            print(f"  - {et['types']}: {et['count']} entities")
        
        # Sample relationships
        result = session.run("MATCH (e1:Entity)-[r]->(e2:Entity) RETURN DISTINCT type(r) as rel_type, count(*) as count ORDER BY count DESC LIMIT 10")
        relationships = list(result)
        
        print(f"\nRelationship types:")
        for rel in relationships:
            print(f"  - {rel['rel_type']}: {rel['count']} relationships")
        
        # Sample entities by name
        result = session.run("MATCH (n:Entity) WHERE n.name IS NOT NULL RETURN n.name as name ORDER BY n.name LIMIT 20")
        entities = list(result)
        
        print(f"\nSample entities (first 20):")
        for entity in entities:
            print(f"  - {entity['name']}")
except Exception as e:
    print(f"Error executing Neo4j queries: {e}")
    driver.close()
    sys.exit(1)

# Ensure driver cleanup
try:
    driver.close()
except Exception as e:
    print(f"Warning: Error closing driver: {e}")
    sys.exit(1)

sys.exit(0)  # Successful execution