#!/usr/bin/env python3
"""Force clear Neo4j database without confirmation."""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

def clear_neo4j():
    """Clear all data from Neo4j."""
    load_dotenv()
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        print("❌ Missing Neo4j password")
        return False
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Delete all relationships first
            result = session.run("MATCH ()-[r]->() DELETE r")
            print(f"✅ Deleted all relationships")
            
            # Then delete all nodes
            result = session.run("MATCH (n) DELETE n")
            print(f"✅ Deleted all nodes")
            
            # Verify
            count_result = session.run("MATCH (n) RETURN count(n) as count")
            count = count_result.single()["count"]
            
            if count == 0:
                print(f"✅ Neo4j database cleared successfully")
                return True
            else:
                print(f"⚠️  {count} nodes still remain")
                return False
        
    except Exception as e:
        print(f"❌ Error clearing Neo4j: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    print("Clearing Neo4j database...")
    success = clear_neo4j()
    exit(0 if success else 1)