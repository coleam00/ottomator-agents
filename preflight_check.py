#!/usr/bin/env python3
"""Pre-flight checks for Neo4j bulk ingestion."""
import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from neo4j import GraphDatabase
import json
from datetime import datetime

def check_supabase():
    """Check Supabase connection and document count."""
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_service_key:
        return False, "Missing Supabase configuration"
    
    try:
        supabase: Client = create_client(supabase_url, supabase_service_key)
        
        # Count documents
        doc_response = supabase.table("documents").select("*", count="exact").execute()
        doc_count = doc_response.count if hasattr(doc_response, 'count') else len(doc_response.data)
        
        # Count chunks
        chunk_response = supabase.table("chunks").select("*", count="exact").execute()
        chunk_count = chunk_response.count if hasattr(chunk_response, 'count') else len(chunk_response.data)
        
        # Get document names
        docs = supabase.table("documents").select("title").execute()
        doc_titles = [d['title'] for d in docs.data] if docs.data else []
        
        return True, {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "document_titles": doc_titles
        }
        
    except Exception as e:
        return False, f"Supabase error: {e}"

def check_neo4j():
    """Check Neo4j connection and existing data."""
    load_dotenv()
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        return False, "Missing Neo4j password"
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]
            
            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            
            # Count episodes
            episode_result = session.run("MATCH (e:EpisodeNode) RETURN count(e) as count")
            episode_count = episode_result.single()["count"]
            
            # Count entities
            entity_result = session.run("MATCH (e:EntityNode) RETURN count(e) as count")
            entity_count = entity_result.single()["count"]
            
            # Get sample of existing data
            sample_result = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            node_types = []
            for record in sample_result:
                node_types.append({
                    "labels": record["labels"],
                    "count": record["count"]
                })
            
        driver.close()
        
        return True, {
            "total_nodes": node_count,
            "total_relationships": rel_count,
            "episode_nodes": episode_count,
            "entity_nodes": entity_count,
            "node_types": node_types
        }
        
    except Exception as e:
        return False, f"Neo4j error: {e}"

def check_script():
    """Check if bulk ingestion script exists."""
    script_path = "/Users/kikocoelho/Documents/Development/MaryPause_AI/ottomator-agents/neo4j_bulk_ingestion.py"
    
    if os.path.exists(script_path):
        # Get file info
        file_stat = os.stat(script_path)
        file_size = file_stat.st_size
        
        return True, {
            "script_exists": True,
            "script_path": script_path,
            "script_size": f"{file_size:,} bytes"
        }
    else:
        return False, "Bulk ingestion script not found"

def main():
    """Run all pre-flight checks."""
    print("=" * 60)
    print("NEO4J BULK INGESTION PRE-FLIGHT CHECKS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Check Supabase
    print("1. SUPABASE CHECK")
    print("-" * 40)
    success, result = check_supabase()
    if success:
        print(f"✅ Supabase connected")
        print(f"   Documents: {result['document_count']}")
        print(f"   Chunks: {result['chunk_count']}")
        if result['document_count'] == 11:
            print(f"   ✅ All 11 documents present")
        else:
            print(f"   ⚠️  Expected 11 documents, found {result['document_count']}")
        print(f"\n   Document titles:")
        for title in result['document_titles']:
            print(f"   - {title}")
    else:
        print(f"❌ {result}")
    
    print("\n2. NEO4J CHECK")
    print("-" * 40)
    success, result = check_neo4j()
    if success:
        print(f"✅ Neo4j connected")
        print(f"   Total nodes: {result['total_nodes']:,}")
        print(f"   Total relationships: {result['total_relationships']:,}")
        print(f"   Episode nodes: {result['episode_nodes']:,}")
        print(f"   Entity nodes: {result['entity_nodes']:,}")
        
        if result['total_nodes'] > 0:
            print(f"\n   ⚠️  Neo4j contains existing data")
            print(f"   Node types in database:")
            for node_type in result['node_types']:
                print(f"   - {node_type['labels']}: {node_type['count']:,}")
            print(f"\n   Consider clearing with: python clean_all_databases.py --neo4j-only")
    else:
        print(f"❌ {result}")
    
    print("\n3. SCRIPT CHECK")
    print("-" * 40)
    success, result = check_script()
    if success:
        print(f"✅ Bulk ingestion script ready")
        print(f"   Path: {result['script_path']}")
        print(f"   Size: {result['script_size']}")
    else:
        print(f"❌ {result}")
    
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()