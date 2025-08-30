#!/usr/bin/env python
"""
Quick vector-only ingestion script for faster testing.
Skips knowledge graph building to focus on vector embeddings.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run vector-only ingestion."""
    logger.info("🚀 Starting quick vector-only ingestion (no knowledge graph)")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    try:
        # Import ingestion module with no graph flag
        import subprocess
        
        # Run ingestion without knowledge graph (--fast mode)
        logger.info("Running ingestion without knowledge graph (--fast mode)...")
        result = subprocess.run(
            ["python", "-m", "ingestion.ingest", "--clean", "--fast", "--no-semantic"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Ingestion failed: {result.stderr}")
            return 1
        
        logger.info(result.stdout)
        
        # Verify results
        from agent.supabase_db_utils import supabase_pool
        supabase = supabase_pool.initialize()
        
        doc_count = supabase.table("documents").select("count", count="exact").execute()
        chunk_count = supabase.table("chunks").select("count", count="exact").execute()
        
        # Sample a chunk to verify dimension
        sample_chunk = supabase.table("chunks").select("embedding").limit(1).execute()
        embedding_dim = None
        if sample_chunk.data and sample_chunk.data[0]["embedding"]:
            embedding_dim = len(sample_chunk.data[0]["embedding"])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("INGESTION SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Documents: {doc_count.count}")
        logger.info(f"✅ Chunks: {chunk_count.count}")
        logger.info(f"✅ Embedding Dimension: {embedding_dim}")
        logger.info(f"⏱️ Duration: {duration:.1f} seconds")
        logger.info("="*60)
        
        if embedding_dim == 768 and doc_count.count == 11:
            logger.info("\n🎉 SUCCESS! Vector ingestion completed with correct dimensions!")
            return 0
        else:
            logger.error(f"\n❌ Issues detected: Expected 11 docs with 768-dim embeddings")
            return 1
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)