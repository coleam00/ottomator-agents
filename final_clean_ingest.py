#!/usr/bin/env python
"""
Final cleanup and re-ingestion with verified 768-dimension embeddings.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def clean_supabase():
    """Clean all data from Supabase."""
    logger.info("🧹 Cleaning Supabase database...")
    
    from agent.supabase_db_utils import supabase_pool
    supabase = supabase_pool.initialize()
    
    try:
        # Delete in correct order due to foreign keys
        tables = ["messages", "sessions", "chunks", "documents"]
        
        for table in tables:
            logger.info(f"  Deleting all records from {table}...")
            result = supabase.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            count = len(result.data) if result.data else 0
            logger.info(f"    ✅ Deleted {count} records from {table}")
        
        # Verify empty
        doc_count = supabase.table("documents").select("count", count="exact").execute()
        chunk_count = supabase.table("chunks").select("count", count="exact").execute()
        
        if doc_count.count == 0 and chunk_count.count == 0:
            logger.info("  ✅ Supabase cleanup complete - all tables empty")
            return True
        else:
            logger.error(f"  ❌ Cleanup incomplete: {doc_count.count} docs, {chunk_count.count} chunks remain")
            return False
            
    except Exception as e:
        logger.error(f"  ❌ Cleanup failed: {e}")
        return False


async def verify_embedding_config():
    """Verify embedding configuration."""
    logger.info("🔍 Verifying embedding configuration...")
    
    from ingestion.embedding_truncator import get_target_dimension
    
    target_dim = get_target_dimension()
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    model = os.getenv("EMBEDDING_MODEL")
    
    logger.info(f"  Provider: {provider}")
    logger.info(f"  Model: {model or 'default'}")
    logger.info(f"  Target dimension: {target_dim}")
    
    if target_dim != 768:
        logger.error(f"  ❌ Invalid target dimension: {target_dim} (expected 768)")
        return False
    
    logger.info("  ✅ Configuration verified")
    return True


async def run_ingestion():
    """Run ingestion with fast mode (no knowledge graph)."""
    logger.info("📥 Running ingestion (fast mode, no knowledge graph)...")
    
    start_time = datetime.now()
    
    # Run ingestion subprocess
    result = subprocess.run(
        ["python", "-m", "ingestion.ingest", "--clean", "--fast", "--no-semantic", "--verbose"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"  ❌ Ingestion failed: {result.stderr}")
        return False
    
    # Log output
    for line in result.stdout.split('\n'):
        if line.strip():
            logger.info(f"  {line}")
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"  ⏱️ Ingestion completed in {duration:.1f} seconds")
    return True


async def verify_results():
    """Verify ingestion results."""
    logger.info("🔍 Verifying results...")
    
    from agent.supabase_db_utils import supabase_pool
    supabase = supabase_pool.initialize()
    
    # Check counts
    doc_count = supabase.table("documents").select("count", count="exact").execute()
    chunk_count = supabase.table("chunks").select("count", count="exact").execute()
    
    logger.info(f"  Documents: {doc_count.count}")
    logger.info(f"  Chunks: {chunk_count.count}")
    
    # Check embedding dimensions
    sample_chunks = supabase.table("chunks").select("embedding").limit(10).execute()
    
    if sample_chunks.data:
        dimensions = set()
        for chunk in sample_chunks.data:
            if chunk.get('embedding'):
                dimensions.add(len(chunk['embedding']))
        
        logger.info(f"  Embedding dimensions found: {sorted(dimensions)}")
        
        if dimensions == {768}:
            logger.info("  ✅ All embeddings are 768 dimensions")
            return True
        else:
            logger.error(f"  ❌ Invalid embedding dimensions: {dimensions}")
            return False
    else:
        logger.error("  ❌ No chunks found")
        return False


async def main():
    """Main execution."""
    logger.info("=" * 60)
    logger.info("FINAL CLEANUP AND RE-INGESTION")
    logger.info("=" * 60)
    
    # Step 1: Verify configuration
    if not await verify_embedding_config():
        logger.error("Configuration verification failed")
        return 1
    
    # Step 2: Clean database
    if not await clean_supabase():
        logger.error("Database cleanup failed")
        return 1
    
    # Step 3: Run ingestion
    if not await run_ingestion():
        logger.error("Ingestion failed")
        return 1
    
    # Step 4: Verify results
    if not await verify_results():
        logger.error("Result verification failed")
        return 1
    
    logger.info("=" * 60)
    logger.info("🎉 SUCCESS! Database cleaned and re-ingested with 768-dim embeddings")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)