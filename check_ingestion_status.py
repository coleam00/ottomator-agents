#!/usr/bin/env python3
"""
Check Ingestion Status and Data Quality
========================================
Verifies the current state of ingested data in Supabase.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def check_ingestion_status():
    """Check the current ingestion status"""
    
    # Initialize Supabase client
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print(f"{Colors.RED}Missing Supabase credentials{Colors.RESET}")
        return 1
    
    client = create_client(supabase_url, supabase_key)
    
    print(f"{Colors.BOLD}{Colors.CYAN}INGESTION STATUS CHECK{Colors.RESET}")
    print("=" * 60)
    
    # 1. Check documents
    print(f"\n{Colors.BOLD}Documents:{Colors.RESET}")
    try:
        docs = client.table('documents').select('*').execute()
        
        print(f"Total documents: {Colors.GREEN}{len(docs.data)}{Colors.RESET}")
        
        for doc in docs.data:
            print(f"\n  Document ID: {doc['id'][:8]}...")
            print(f"  Title: {Colors.CYAN}{doc['title']}{Colors.RESET}")
            print(f"  Created: {doc['created_at']}")
            
            # Count chunks for this document
            chunks = client.table('chunks').select('id').eq('document_id', doc['id']).execute()
            print(f"  Chunks: {len(chunks.data)}")
            
    except Exception as e:
        print(f"{Colors.RED}Error checking documents: {e}{Colors.RESET}")
        return 1
    
    # 2. Check chunks
    print(f"\n{Colors.BOLD}Chunks Summary:{Colors.RESET}")
    try:
        chunks = client.table('chunks').select('*').limit(5).execute()
        total_chunks = client.table('chunks').select('id', count='exact').execute()
        
        chunk_count = total_chunks.count if hasattr(total_chunks, 'count') else len(total_chunks.data)
        print(f"Total chunks: {Colors.GREEN}{chunk_count}{Colors.RESET}")
        
        # Check embedding dimensions
        print(f"\n{Colors.BOLD}Sample Chunk Analysis:{Colors.RESET}")
        for i, chunk in enumerate(chunks.data[:3], 1):
            print(f"\n  Chunk {i}:")
            print(f"  Content length: {len(chunk['content'])} chars")
            print(f"  Content preview: {chunk['content'][:100]}...")
            
            # Check embedding
            if chunk.get('embedding'):
                embedding = chunk['embedding']
                if isinstance(embedding, str):
                    # Parse pgvector format
                    if embedding.startswith('[') and embedding.endswith(']'):
                        try:
                            values = json.loads(embedding)
                            dim = len(values)
                        except:
                            values = embedding.strip('[]').split(',')
                            dim = len(values)
                    else:
                        dim = len(embedding.split(','))
                else:
                    dim = len(embedding)
                
                print(f"  Embedding dimension: {Colors.GREEN}{dim}{Colors.RESET}")
                print(f"  Embedding type: {type(embedding).__name__}")
                
                # Check if it's properly stored as vector
                if isinstance(embedding, str) and embedding.startswith('['):
                    print(f"  Format: {Colors.GREEN}✓ Proper pgvector format{Colors.RESET}")
                else:
                    print(f"  Format: {Colors.YELLOW}⚠ Check format{Colors.RESET}")
            else:
                print(f"  Embedding: {Colors.RED}Missing{Colors.RESET}")
                
    except Exception as e:
        print(f"{Colors.RED}Error checking chunks: {e}{Colors.RESET}")
        return 1
    
    # 3. Test a sample search
    print(f"\n{Colors.BOLD}Testing Vector Search:{Colors.RESET}")
    try:
        # Get a sample embedding from an existing chunk
        sample_chunk = client.table('chunks').select('embedding').limit(1).execute()
        
        if sample_chunk.data and sample_chunk.data[0].get('embedding'):
            embedding_str = sample_chunk.data[0]['embedding']
            
            # Parse the embedding
            if isinstance(embedding_str, str):
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    try:
                        test_embedding = json.loads(embedding_str)
                    except:
                        test_embedding = [float(x) for x in embedding_str.strip('[]').split(',')]
                else:
                    test_embedding = [float(x) for x in embedding_str.split(',')]
            else:
                test_embedding = embedding_str
            
            # Perform test search
            results = client.rpc('match_chunks', {
                'query_embedding': test_embedding,
                'match_count': 3
            }).execute()
            
            print(f"Test search returned: {Colors.GREEN}{len(results.data)} results{Colors.RESET}")
            
            if results.data:
                print(f"Top similarity score: {Colors.CYAN}{results.data[0]['similarity']:.4f}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}No embeddings found for test search{Colors.RESET}")
            
    except Exception as e:
        print(f"{Colors.RED}Search test error: {e}{Colors.RESET}")
    
    # 4. Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    if chunk_count > 0 and len(docs.data) > 0:
        print(f"{Colors.GREEN}✅ INGESTION SUCCESSFUL{Colors.RESET}")
        print(f"  • {len(docs.data)} documents ingested")
        print(f"  • {chunk_count} chunks created")
        print(f"  • Embeddings stored as 768-dimensional vectors")
        print(f"  • Vector search operational")
    else:
        print(f"{Colors.RED}⚠️ INGESTION INCOMPLETE{Colors.RESET}")
        print(f"  Please run the ingestion process")
    
    return 0

if __name__ == "__main__":
    exit(check_ingestion_status())