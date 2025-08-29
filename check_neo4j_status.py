#!/usr/bin/env python
"""Script to check Neo4j database status and content."""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def check_neo4j_status():
    """Check the status and content of Neo4j database."""
    
    print("=" * 50)
    print("NEO4J DATABASE STATUS CHECK")
    print("=" * 50)
    print(f"URI: {NEO4J_URI}")
    print(f"User: {NEO4J_USER}")
    print("-" * 50)
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # 1. Count all nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            total_nodes = result.single()["count"]
            print(f"\nTotal nodes in database: {total_nodes}")
            
            # 2. Count Entity nodes
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            entity_count = result.single()["count"]
            print(f"Entity nodes: {entity_count}")
            
            # 3. Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            relationship_count = result.single()["count"]
            print(f"Total relationships: {relationship_count}")
            
            # 4. Check for episodic memories
            result = session.run("MATCH (e:EpisodicMemory) RETURN count(e) as count")
            episodic_count = result.single()["count"]
            print(f"Episodic memories: {episodic_count}")
            
            # 5. List all node labels
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            print(f"\nNode labels in database: {labels if labels else 'None'}")
            
            # 6. If there are entities, show a sample
            if entity_count > 0:
                print("\nSample entities (first 5):")
                result = session.run("MATCH (n:Entity) RETURN n.name as name LIMIT 5")
                for record in result:
                    print(f"  - {record['name']}")
            
            # 7. Check for any nodes at all (in case they're not labeled properly)
            if total_nodes > 0 and entity_count == 0:
                print("\nChecking for nodes without Entity label:")
                result = session.run("MATCH (n) RETURN labels(n) as labels, count(*) as count GROUP BY labels(n) LIMIT 10")
                for record in result:
                    print(f"  - Labels: {record['labels']}, Count: {record['count']}")
        
        driver.close()
        
        print("\n" + "=" * 50)
        if total_nodes == 0:
            print("❌ Neo4j database is EMPTY - no data has been ingested")
        else:
            print(f"✅ Neo4j contains {total_nodes} nodes and {relationship_count} relationships")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error connecting to Neo4j: {e}")
        print("Please check your Neo4j credentials and connection settings.")
        
if __name__ == "__main__":
    check_neo4j_status()