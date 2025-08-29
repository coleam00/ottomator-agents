#!/usr/bin/env python3
"""
Fast ingestion script that skips semantic chunking and graph building.
This will quickly get all documents into the vector database.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def main():
    """Run fast ingestion without semantic chunking."""
    try:
        # Clean databases first
        logger.info("=" * 60)
        logger.info("STARTING FAST INGESTION")
        logger.info("=" * 60)
        
        # First clean the databases
        logger.info("Step 1: Cleaning databases...")
        os.system("python clean_all_databases_force.py")
        
        # Wait a moment for cleanup to complete
        await asyncio.sleep(2)
        
        # Run ingestion without semantic chunking and without graph building
        logger.info("\nStep 2: Running fast ingestion (no semantic chunking, no graph)...")
        logger.info("This will quickly load all documents into the vector database.")
        
        # Run the ingestion with specific flags
        cmd = "python -m ingestion.ingest --no-semantic --no-graph --verbose"
        logger.info(f"Running: {cmd}")
        
        result = os.system(cmd)
        
        if result == 0:
            logger.info("\n✅ Fast ingestion completed successfully!")
            
            # Check the results
            logger.info("\nStep 3: Checking ingestion results...")
            os.system("python check_ingestion_status.py")
        else:
            logger.error(f"\n❌ Ingestion failed with exit code: {result}")
            
    except Exception as e:
        logger.error(f"Error during fast ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())