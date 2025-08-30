#!/usr/bin/env python
"""Test the knowledge base search functionality after Neo4j ingestion."""

import asyncio
import os
from dotenv import load_dotenv
from agent.graph_utils import GraphitiClient
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_knowledge_base_search():
    """Test various knowledge base search queries."""
    
    print("=" * 60)
    print("TESTING KNOWLEDGE BASE SEARCH")
    print("=" * 60)
    
    # Initialize the graph client
    graph_client = GraphitiClient()
    await graph_client.initialize()
    
    try:
        # Test searches
        test_queries = [
            "menopause symptoms",
            "estrogen therapy",
            "mindfulness meditation",
            "hot flashes treatment",
            "perimenopause vs menopause"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing search: '{query}'")
            print("-" * 40)
            
            try:
                # Test search (which uses the graph)
                results = await graph_client.search(
                    query=query,
                    group_ids=["0"],  # Shared knowledge base
                    use_hybrid_search=True
                )
                
                if results:
                    print(f"✅ Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        # Handle different result types (results are dictionaries)
                        if isinstance(result, dict):
                            fact = result.get('fact', 'N/A')
                            print(f"  {i}. Fact: {fact[:100] if fact else 'N/A'}...")
                        else:
                            print(f"  {i}. Result: {str(result)[:100]}...")
                else:
                    print("❌ No results found")
                    
            except Exception as e:
                print(f"❌ Search failed: {e}")
        
        # Test entity search specifically
        print("\n" + "=" * 60)
        print("TESTING ENTITY RELATIONSHIPS")
        print("=" * 60)
        
        test_entities = ["menopause", "estrogen", "mindfulness"]
        
        for entity in test_entities:
            print(f"\n🔍 Getting relationships for: '{entity}'")
            print("-" * 40)
            
            try:
                # Search for the entity first
                search_results = await graph_client.search(
                    query=entity,
                    group_ids=["0"],
                    use_hybrid_search=True
                )
                
                if search_results:
                    print(f"✅ Found entity in graph")
                    # Note: Getting relationships might require direct Neo4j queries
                    # since GraphitiClient may not have a direct method for this
                else:
                    print(f"❌ Entity not found in graph")
                    
            except Exception as e:
                print(f"❌ Failed to get relationships: {e}")
        
        # Test direct graph statistics
        print("\n" + "=" * 60)
        print("GRAPH STATISTICS")
        print("=" * 60)
        
        # Import Neo4j driver for direct queries
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        
        with driver.session() as session:
            # Count entities by category
            result = session.run("""
                MATCH (n:Entity)
                WHERE n.name CONTAINS 'menopause' OR n.name CONTAINS 'estrogen' 
                   OR n.name CONTAINS 'symptom' OR n.name CONTAINS 'treatment'
                RETURN n.name as name
                ORDER BY n.name
                LIMIT 20
            """)
            
            medical_entities = list(result)
            
            print(f"\n📊 Medical entities in graph:")
            for entity in medical_entities:
                print(f"  - {entity['name']}")
            
            # Count relationships
            result = session.run("""
                MATCH (e1:Entity)-[r]->(e2:Entity)
                WHERE e1.name CONTAINS 'menopause' OR e2.name CONTAINS 'menopause'
                RETURN e1.name as source, type(r) as rel_type, e2.name as target
                LIMIT 10
            """)
            
            relationships = list(result)
            
            print(f"\n📊 Sample menopause-related relationships:")
            for rel in relationships:
                print(f"  - {rel['source']} --[{rel['rel_type']}]--> {rel['target']}")
        
        driver.close()
        
        print("\n" + "=" * 60)
        print("✅ KNOWLEDGE BASE SEARCH TESTING COMPLETE")
        print("=" * 60)
        
    finally:
        await graph_client.close()

if __name__ == "__main__":
    asyncio.run(test_knowledge_base_search())