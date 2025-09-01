#!/usr/bin/env python3
"""
Apply database fixes for hybrid_search type mismatch.

This script connects to Supabase and applies the necessary SQL migration
to fix the type mismatch error in hybrid_search function.
"""

import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_supabase_client() -> Client:
    """Get Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    
    return create_client(url, key)

def apply_migration():
    """Apply the database migration to fix hybrid_search."""
    try:
        # Read the migration file
        migration_path = "sql/fix_hybrid_search_types.sql"
        if not os.path.exists(migration_path):
            logger.error(f"Migration file not found: {migration_path}")
            return False
        
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        logger.info("=" * 60)
        logger.info("DATABASE MIGRATION - Fix Hybrid Search Types")
        logger.info("=" * 60)
        
        # Split the migration into individual statements
        statements = [s.strip() for s in migration_sql.split(';') if s.strip() and not s.strip().startswith('--')]
        
        client = get_supabase_client()
        
        logger.info(f"\nExecuting {len(statements)} SQL statements...")
        
        success_count = 0
        for i, statement in enumerate(statements, 1):
            try:
                # Add semicolon back
                statement = statement + ';'
                
                # Show what we're executing (first 100 chars)
                preview = statement[:100].replace('\n', ' ')
                if len(statement) > 100:
                    preview += "..."
                logger.info(f"\n{i}. Executing: {preview}")
                
                # Execute via Supabase RPC (we'll create a helper function)
                # Note: Direct SQL execution through Supabase client is limited
                # You may need to run this in Supabase SQL Editor instead
                
                logger.warning("Note: Direct SQL execution through Supabase Python client is limited.")
                logger.warning("Please run the migration in Supabase SQL Editor instead.")
                logger.info(f"\nMigration file location: {os.path.abspath(migration_path)}")
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to execute statement {i}: {e}")
                continue
        
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION INSTRUCTIONS")
        logger.info("=" * 60)
        logger.info("\nSince direct SQL execution is limited through the Python client,")
        logger.info("please follow these steps to apply the migration:\n")
        logger.info("1. Go to your Supabase Dashboard")
        logger.info("2. Navigate to SQL Editor")
        logger.info("3. Create a new query")
        logger.info(f"4. Copy and paste the contents of: {os.path.abspath(migration_path)}")
        logger.info("5. Click 'Run' to execute the migration")
        logger.info("\nThis will fix the hybrid_search type mismatch error.")
        
        # Test if the issue exists
        logger.info("\n" + "=" * 60)
        logger.info("TESTING CURRENT STATE")
        logger.info("=" * 60)
        
        try:
            # Try to call hybrid_search to see if it works
            from agent.unified_db_utils import get_db_utils
            import asyncio
            
            async def test_hybrid_search():
                db = get_db_utils()
                # Create a dummy embedding
                dummy_embedding = [0.1] * 768
                try:
                    results = await db.hybrid_search(
                        query_embedding=dummy_embedding,
                        query_text="test",
                        limit=1
                    )
                    logger.info("✅ hybrid_search is working correctly!")
                    return True
                except Exception as e:
                    if "does not match expected type" in str(e):
                        logger.error("❌ hybrid_search has type mismatch error - migration needed!")
                    else:
                        logger.warning(f"⚠️ hybrid_search error (may be unrelated): {e}")
                    return False
            
            # Run the test
            works = asyncio.run(test_hybrid_search())
            
            if works:
                logger.info("\n✅ No migration needed - hybrid_search is already working!")
            else:
                logger.info("\n❌ Migration needed - please apply it in Supabase SQL Editor")
                
        except Exception as e:
            logger.error(f"Could not test hybrid_search: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

def main():
    """Main entry point."""
    try:
        success = apply_migration()
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✅ Migration instructions provided successfully!")
            logger.info("=" * 60)
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ Migration needs manual application")
            logger.error("=" * 60)
            
    except Exception as e:
        logger.error(f"Failed to apply migration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())