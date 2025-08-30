#!/usr/bin/env python
"""
Complete database cleanup and re-ingestion script.

This script:
1. Cleans both Supabase and Neo4j databases completely
2. Verifies databases are empty
3. Runs fresh ingestion with proper 768-dimension embeddings
4. Monitors progress and provides statistics
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.ERROR)


async def cleanup_supabase() -> Dict[str, Any]:
    """Clean all records from Supabase tables."""
    logger.info("🧹 Starting Supabase cleanup...")
    
    results = {
        "documents": 0,
        "chunks": 0,
        "sessions": 0,
        "messages": 0,
        "errors": []
    }
    
    try:
        from agent.supabase_db_utils import supabase_pool
        
        # Initialize and get Supabase client
        supabase = supabase_pool.initialize()
        
        # Delete all messages first (foreign key constraint)
        logger.info("  Deleting messages...")
        try:
            msg_result = supabase.table("messages").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            results["messages"] = len(msg_result.data) if msg_result.data else 0
            logger.info(f"  ✅ Deleted {results['messages']} messages")
        except Exception as e:
            logger.error(f"  ❌ Failed to delete messages: {e}")
            results["errors"].append(f"messages: {e}")
        
        # Delete all sessions
        logger.info("  Deleting sessions...")
        try:
            sess_result = supabase.table("sessions").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            results["sessions"] = len(sess_result.data) if sess_result.data else 0
            logger.info(f"  ✅ Deleted {results['sessions']} sessions")
        except Exception as e:
            logger.error(f"  ❌ Failed to delete sessions: {e}")
            results["errors"].append(f"sessions: {e}")
        
        # Delete all chunks (foreign key constraint)
        logger.info("  Deleting chunks...")
        try:
            chunk_result = supabase.table("chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            results["chunks"] = len(chunk_result.data) if chunk_result.data else 0
            logger.info(f"  ✅ Deleted {results['chunks']} chunks")
        except Exception as e:
            logger.error(f"  ❌ Failed to delete chunks: {e}")
            results["errors"].append(f"chunks: {e}")
        
        # Delete all documents
        logger.info("  Deleting documents...")
        try:
            doc_result = supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            results["documents"] = len(doc_result.data) if doc_result.data else 0
            logger.info(f"  ✅ Deleted {results['documents']} documents")
        except Exception as e:
            logger.error(f"  ❌ Failed to delete documents: {e}")
            results["errors"].append(f"documents: {e}")
        
        # Verify deletion
        logger.info("  Verifying cleanup...")
        doc_count = supabase.table("documents").select("count", count="exact").execute()
        chunk_count = supabase.table("chunks").select("count", count="exact").execute()
        sess_count = supabase.table("sessions").select("count", count="exact").execute()
        msg_count = supabase.table("messages").select("count", count="exact").execute()
        
        actual_counts = {
            "documents": doc_count.count if doc_count else 0,
            "chunks": chunk_count.count if chunk_count else 0,
            "sessions": sess_count.count if sess_count else 0,
            "messages": msg_count.count if msg_count else 0
        }
        
        if all(count == 0 for count in actual_counts.values()):
            logger.info("  ✅ Supabase cleanup complete - all tables empty")
        else:
            logger.warning(f"  ⚠️ Some records remain: {actual_counts}")
            results["errors"].append(f"Remaining records: {actual_counts}")
        
    except Exception as e:
        logger.error(f"Supabase cleanup failed: {e}")
        results["errors"].append(str(e))
    
    return results


async def cleanup_neo4j() -> Dict[str, Any]:
    """Clean all nodes and relationships from Neo4j."""
    logger.info("🧹 Starting Neo4j cleanup...")
    
    results = {
        "nodes_deleted": 0,
        "relationships_deleted": 0,
        "errors": []
    }
    
    try:
        from agent.graph_utils import graph_client
        
        # Initialize graph client
        await graph_client.initialize()
        
        # Clear all data using Graphiti's clear_data function
        logger.info("  Clearing all graph data...")
        await graph_client.clear_graph()
        
        # Get statistics to verify
        stats = await graph_client.get_graph_statistics()
        logger.info(f"  Graph statistics after cleanup: {stats}")
        
        # Close the client
        await graph_client.close()
        
        logger.info("  ✅ Neo4j cleanup complete")
        
    except Exception as e:
        logger.error(f"Neo4j cleanup failed: {e}")
        results["errors"].append(str(e))
    
    return results


async def verify_embedding_config() -> bool:
    """Verify embedding configuration is set to 768 dimensions."""
    logger.info("🔍 Verifying embedding configuration...")
    
    try:
        from ingestion.embedding_truncator import get_target_dimension
        from agent.models import _safe_parse_int
        
        # Check target dimension
        target_dim = get_target_dimension()
        logger.info(f"  Target embedding dimension: {target_dim}")
        
        if target_dim != 768:
            logger.error(f"  ❌ Expected 768 dimensions, got {target_dim}")
            return False
        
        # Check environment variables
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        
        logger.info(f"  Embedding provider: {embedding_provider}")
        logger.info(f"  Embedding model: {embedding_model or 'default'}")
        
        # Verify provider/model compatibility
        if embedding_provider in ("gemini", "google"):
            expected_model = "gemini-embedding-001"
        else:
            expected_model = "text-embedding-3-small"
        
        if not embedding_model:
            logger.info(f"  Using default model for {embedding_provider}: {expected_model}")
        elif embedding_model != expected_model:
            logger.warning(f"  ⚠️ Model {embedding_model} may not be optimal for provider {embedding_provider}")
        
        logger.info("  ✅ Embedding configuration verified")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify embedding configuration: {e}")
        return False


async def run_ingestion() -> Dict[str, Any]:
    """Run fresh ingestion of all medical documents."""
    logger.info("📥 Starting document ingestion...")
    
    results = {
        "documents_processed": 0,
        "chunks_created": 0,
        "graph_episodes": 0,
        "errors": [],
        "duration_seconds": 0
    }
    
    start_time = datetime.now()
    
    try:
        # Import ingestion module
        from ingestion.ingest import main as ingest_main
        
        # Run ingestion with monitoring
        logger.info("  Running ingestion process...")
        logger.info("  This may take 5-10 minutes for semantic chunking...")
        
        # Create a simple progress indicator
        async def show_progress():
            """Show progress dots while ingestion runs."""
            count = 0
            while True:
                await asyncio.sleep(10)
                count += 1
                logger.info(f"  ... still processing ({count * 10} seconds elapsed)")
        
        # Run ingestion with progress indicator
        progress_task = asyncio.create_task(show_progress())
        
        try:
            # Run the actual ingestion
            await ingest_main()
            
            # Cancel progress indicator
            progress_task.cancel()
            
            # Get statistics from database
            from agent.supabase_db_utils import supabase_pool
            supabase = supabase_pool.initialize()
            
            doc_count = supabase.table("documents").select("count", count="exact").execute()
            chunk_count = supabase.table("chunks").select("count", count="exact").execute()
            
            results["documents_processed"] = doc_count.count if doc_count else 0
            results["chunks_created"] = chunk_count.count if chunk_count else 0
            
            # Check Neo4j
            from agent.graph_utils import graph_client
            await graph_client.initialize()
            stats = await graph_client.get_graph_statistics()
            await graph_client.close()
            
            logger.info(f"  ✅ Ingestion complete!")
            logger.info(f"    - Documents: {results['documents_processed']}")
            logger.info(f"    - Chunks: {results['chunks_created']}")
            logger.info(f"    - Graph status: {stats}")
            
        except asyncio.CancelledError:
            pass  # Expected when cancelling progress task
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        results["errors"].append(str(e))
    
    end_time = datetime.now()
    results["duration_seconds"] = (end_time - start_time).total_seconds()
    
    return results


async def verify_final_state() -> Dict[str, Any]:
    """Verify final state of both databases."""
    logger.info("🔍 Verifying final database state...")
    
    results = {
        "supabase": {},
        "neo4j": {},
        "success": False
    }
    
    try:
        # Check Supabase
        from agent.supabase_db_utils import supabase_pool
        supabase = supabase_pool.initialize()
        
        doc_count = supabase.table("documents").select("count", count="exact").execute()
        chunk_count = supabase.table("chunks").select("count", count="exact").execute()
        
        results["supabase"] = {
            "documents": doc_count.count if doc_count else 0,
            "chunks": chunk_count.count if chunk_count else 0
        }
        
        # Sample a chunk to verify dimension
        sample_chunk = supabase.table("chunks").select("embedding").limit(1).execute()
        if sample_chunk.data and sample_chunk.data[0]["embedding"]:
            embedding_dim = len(sample_chunk.data[0]["embedding"])
            results["supabase"]["embedding_dimension"] = embedding_dim
            
            if embedding_dim != 768:
                logger.error(f"  ❌ Embedding dimension mismatch: {embedding_dim} != 768")
            else:
                logger.info(f"  ✅ Embedding dimension correct: 768")
        
        # Check Neo4j
        from agent.graph_utils import graph_client
        await graph_client.initialize()
        stats = await graph_client.get_graph_statistics()
        await graph_client.close()
        
        results["neo4j"] = stats
        
        # Determine success
        if results["supabase"]["documents"] == 11 and results["supabase"]["chunks"] > 0:
            if results["supabase"].get("embedding_dimension") == 768:
                results["success"] = True
                logger.info("  ✅ All checks passed!")
            else:
                logger.error("  ❌ Embedding dimension check failed")
        else:
            logger.error(f"  ❌ Document/chunk count check failed")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FINAL STATE SUMMARY")
        logger.info("="*60)
        logger.info(f"Supabase:")
        logger.info(f"  - Documents: {results['supabase']['documents']}")
        logger.info(f"  - Chunks: {results['supabase']['chunks']}")
        logger.info(f"  - Embedding Dimension: {results['supabase'].get('embedding_dimension', 'N/A')}")
        logger.info(f"Neo4j:")
        logger.info(f"  - Status: {results['neo4j']}")
        logger.info(f"Overall Success: {'✅ YES' if results['success'] else '❌ NO'}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Failed to verify final state: {e}")
        results["error"] = str(e)
    
    return results


async def main():
    """Main execution function."""
    logger.info("🚀 Starting complete database cleanup and re-ingestion")
    logger.info("="*60)
    
    # Step 1: Verify embedding configuration
    if not await verify_embedding_config():
        logger.error("❌ Embedding configuration verification failed. Please fix configuration first.")
        return 1
    
    # Step 2: Clean Supabase
    supabase_results = await cleanup_supabase()
    if supabase_results["errors"]:
        logger.error(f"❌ Supabase cleanup had errors: {supabase_results['errors']}")
        # Continue anyway to try Neo4j cleanup
    
    # Step 3: Clean Neo4j
    neo4j_results = await cleanup_neo4j()
    if neo4j_results["errors"]:
        logger.error(f"❌ Neo4j cleanup had errors: {neo4j_results['errors']}")
        # Continue anyway to try ingestion
    
    # Step 4: Run fresh ingestion
    ingestion_results = await run_ingestion()
    if ingestion_results["errors"]:
        logger.error(f"❌ Ingestion had errors: {ingestion_results['errors']}")
        return 1
    
    logger.info(f"\n⏱️ Ingestion took {ingestion_results['duration_seconds']:.1f} seconds")
    
    # Step 5: Verify final state
    final_state = await verify_final_state()
    
    if final_state["success"]:
        logger.info("\n🎉 SUCCESS! Database cleanup and re-ingestion completed successfully!")
        return 0
    else:
        logger.error("\n❌ FAILED! Some issues remain. Please check the logs above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)