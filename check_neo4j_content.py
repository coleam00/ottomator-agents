#!/usr/bin/env python
"""Check detailed Neo4j content."""

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

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

driver.close()