#!/usr/bin/env python3
"""
Comprehensive ingestion monitoring script with real-time progress tracking
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(text: str, char: str = "="):
    """Print formatted header"""
    width = 80
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")

def print_progress_bar(current: int, total: int, label: str = "", width: int = 50):
    """Print a progress bar"""
    if total == 0:
        percentage = 0
    else:
        percentage = current / total
    
    filled = int(width * percentage)
    bar = "█" * filled + "░" * (width - filled)
    
    print(f"\r{label}: [{bar}] {current}/{total} ({percentage:.1%})", end="", flush=True)

def check_supabase_status():
    """Check Supabase connection and data counts"""
    try:
        from supabase import create_client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        supabase = create_client(supabase_url, supabase_key)
        
        # Count documents and chunks
        docs_result = supabase.table('documents').select('id', count='exact').execute()
        chunks_result = supabase.table('chunks').select('id', count='exact').execute()
        
        doc_count = docs_result.count if hasattr(docs_result, 'count') else len(docs_result.data)
        chunk_count = chunks_result.count if hasattr(chunks_result, 'count') else len(chunks_result.data)
        
        return {
            'connected': True,
            'documents': doc_count,
            'chunks': chunk_count
        }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e),
            'documents': 0,
            'chunks': 0
        }

def check_neo4j_status():
    """Check Neo4j connection and data counts"""
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            entity_count = session.run('MATCH (n:Entity) RETURN count(n) as count').single()['count']
            relation_count = session.run('MATCH ()-[r]->() RETURN count(r) as count').single()['count']
            episode_count = session.run('MATCH (e:EpisodicMemory) RETURN count(e) as count').single()['count']
        driver.close()
        
        return {
            'connected': True,
            'entities': entity_count,
            'relations': relation_count,
            'episodes': episode_count
        }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e),
            'entities': 0,
            'relations': 0,
            'episodes': 0
        }

def monitor_checkpoint_file(checkpoint_path: str = "ingestion_checkpoint.json"):
    """Monitor the checkpoint file for progress updates"""
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                return data.get('progress', {}), data.get('checkpoints', [])
        return {}, []
    except Exception:
        return {}, []

async def run_ingestion_with_monitoring():
    """Run the ingestion process with real-time monitoring"""
    
    print_header("MEDICAL RAG INGESTION MONITOR", "═")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initial database status
    print("\n📊 Initial Database Status:")
    print("-" * 40)
    
    supabase_status = check_supabase_status()
    neo4j_status = check_neo4j_status()
    
    print(f"Supabase: {'✅ Connected' if supabase_status['connected'] else '❌ Disconnected'}")
    print(f"  Documents: {supabase_status['documents']}")
    print(f"  Chunks: {supabase_status['chunks']}")
    
    print(f"\nNeo4j: {'✅ Connected' if neo4j_status['connected'] else '❌ Disconnected'}")
    print(f"  Entities: {neo4j_status['entities']}")
    print(f"  Relations: {neo4j_status['relations']}")
    print(f"  Episodes: {neo4j_status['episodes']}")
    
    # Check for medical documents
    docs_dir = Path("medical_docs")
    doc_files = sorted(docs_dir.glob("*.md"))
    total_docs = len(doc_files)
    
    print(f"\n📁 Found {total_docs} documents to process:")
    for i, doc in enumerate(doc_files, 1):
        print(f"  {i}. {doc.name}")
    
    print_header("Starting Ingestion Process", "-")
    
    # Start the ingestion process
    import subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "ingestion.ingest", "--verbose"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Save the PID to temp directory
    import tempfile
    pid_file = os.path.join(tempfile.gettempdir(), "ingestion.pid")
    with open(pid_file, "w") as f:
        f.write(str(process.pid))
    
    print(f"🚀 Ingestion process started (PID: {process.pid})\n")
    
    # Monitor progress
    last_progress = {}
    last_checkpoint_count = 0
    start_time = time.time()
    
    # Real-time output monitoring
    output_buffer = []
    retry_count = 0
    error_count = 0
    
    while True:
        # Check if process is still running
        poll_status = process.poll()
        
        # Read output
        if process.stdout:
            line = process.stdout.readline()
            if line:
                output_buffer.append(line.strip())
                
                # Track retries and errors
                if "retry" in line.lower():
                    retry_count += 1
                if "error" in line.lower() or "exception" in line.lower():
                    error_count += 1
                
                # Print important lines
                if any(keyword in line.lower() for keyword in ['error', 'retry', 'success', 'completed', 'failed', 'processing']):
                    print(f"  📝 {line.strip()}")
        
        # Check checkpoint progress
        progress, checkpoints = monitor_checkpoint_file()
        
        if progress != last_progress:
            # Progress updated
            completed = progress.get('completed_documents', 0)
            successful = progress.get('successful_documents', 0)
            failed = progress.get('failed_documents', 0)
            total = progress.get('total_documents', total_docs)
            
            print(f"\n📈 Progress Update:")
            print_progress_bar(completed, total, "Documents")
            print(f"\n  ✅ Successful: {successful}")
            print(f"  ❌ Failed: {failed}")
            print(f"  🔄 Retries: {retry_count}")
            print(f"  ⚠️ Errors: {error_count}")
            
            if len(checkpoints) > last_checkpoint_count:
                # New checkpoint added
                latest = checkpoints[-1]
                print(f"\n  📍 Latest: {latest['document']}")
                print(f"     Chunks: {latest.get('chunks', 0)}")
                print(f"     Episodes: {latest.get('episodes', 0)}")
                print(f"     Status: {'✅ Success' if latest.get('success', False) else '❌ Failed'}")
            
            last_progress = progress.copy()
            last_checkpoint_count = len(checkpoints)
        
        # Check database status periodically
        if int(time.time() - start_time) % 30 == 0:  # Every 30 seconds
            supabase_current = check_supabase_status()
            neo4j_current = check_neo4j_status()
            
            if (supabase_current['chunks'] != supabase_status['chunks'] or 
                neo4j_current['entities'] != neo4j_status['entities']):
                
                print(f"\n🔄 Database Update:")
                print(f"  Supabase chunks: {supabase_status['chunks']} → {supabase_current['chunks']}")
                print(f"  Neo4j entities: {neo4j_status['entities']} → {neo4j_current['entities']}")
                
                supabase_status = supabase_current
                neo4j_status = neo4j_current
        
        # Process has ended
        if poll_status is not None:
            break
        
        await asyncio.sleep(1)
    
    # Process completed
    elapsed_time = time.time() - start_time
    
    print_header("Ingestion Process Completed", "-")
    print(f"⏱️ Total Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Exit Code: {poll_status}")
    
    # Final database status
    print("\n📊 Final Database Status:")
    print("-" * 40)
    
    supabase_final = check_supabase_status()
    neo4j_final = check_neo4j_status()
    
    print(f"Supabase:")
    print(f"  Documents: {supabase_final['documents']}")
    print(f"  Chunks: {supabase_final['chunks']}")
    print(f"  Chunks Created: {supabase_final['chunks'] - supabase_status['chunks']}")
    
    print(f"\nNeo4j:")
    print(f"  Entities: {neo4j_final['entities']}")
    print(f"  Relations: {neo4j_final['relations']}")
    print(f"  Episodes: {neo4j_final['episodes']}")
    print(f"  Entities Created: {neo4j_final['entities'] - neo4j_status['entities']}")
    print(f"  Relations Created: {neo4j_final['relations'] - neo4j_status['relations']}")
    
    # Validation summary
    print_header("Validation Summary", "-")
    
    if os.path.exists("ingestion_checkpoint.json"):
        with open("ingestion_checkpoint.json", 'r') as f:
            checkpoint_data = json.load(f)
            
            validation_results = checkpoint_data.get('validation_results', [])
            if validation_results:
                print(f"📋 Validation Results for {len(validation_results)} documents:\n")
                
                for result in validation_results:
                    status = "✅" if result.get('supabase_valid') and result.get('neo4j_valid') else "⚠️"
                    print(f"{status} {result.get('title', 'Unknown')}")
                    print(f"   Document ID: {result.get('document_id')}")
                    print(f"   Supabase: {'✅' if result.get('supabase_valid') else '❌'} ({result.get('chunk_count', 0)} chunks)")
                    print(f"   Neo4j: {'✅' if result.get('neo4j_valid') else '❌'} ({result.get('episode_count', 0)} episodes)")
                    
                    if result.get('errors'):
                        print(f"   ⚠️ Errors: {', '.join(result['errors'])}")
                    if result.get('warnings'):
                        print(f"   ⚡ Warnings: {', '.join(result['warnings'])}")
                    print()
    
    # Generate summary report
    summary = {
        'status': 'SUCCESS' if poll_status == 0 else 'FAILED',
        'start_time': datetime.fromtimestamp(start_time).isoformat(),
        'end_time': datetime.now().isoformat(),
        'elapsed_seconds': elapsed_time,
        'documents_processed': total_docs,
        'supabase': {
            'documents': supabase_final['documents'],
            'chunks': supabase_final['chunks'],
            'chunks_created': supabase_final['chunks'] - supabase_status['chunks']
        },
        'neo4j': {
            'entities': neo4j_final['entities'],
            'relations': neo4j_final['relations'],
            'episodes': neo4j_final['episodes'],
            'entities_created': neo4j_final['entities'] - neo4j_status['entities'],
            'relations_created': neo4j_final['relations'] - neo4j_status['relations']
        },
        'monitoring': {
            'retries': retry_count,
            'errors': error_count
        }
    }
    
    # Save summary report
    with open("ingestion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print_header("✅ INGESTION MONITORING COMPLETE", "═")
    print(f"Summary report saved to: ingestion_summary.json")
    
    return poll_status == 0

async def main():
    """Main entry point"""
    try:
        success = await run_ingestion_with_monitoring()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())