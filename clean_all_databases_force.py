#!/usr/bin/env python3
"""
Force version of database cleanup script - no confirmation required.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from neo4j import GraphDatabase
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DatabaseCleaner:
    def __init__(self):
        # Supabase configuration
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        # Neo4j configuration
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
    def clean_supabase(self):
        """Clean all data from Supabase tables while preserving schema."""
        try:
            logger.info("Connecting to Supabase...")
            supabase: Client = create_client(self.supabase_url, self.supabase_key)
            
            # Clean messages table first (foreign key references sessions)
            logger.info("Cleaning messages table...")
            # For UUID columns, we need to use neq with an impossible UUID value
            result = supabase.table("messages").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            logger.info(f"Deleted messages")
            
            # Clean sessions table
            logger.info("Cleaning sessions table...")
            result = supabase.table("sessions").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            logger.info(f"Deleted sessions")
            
            # Clean chunks table (foreign key references documents)
            logger.info("Cleaning chunks table...")
            result = supabase.table("chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            logger.info(f"Deleted chunks")
            
            # Clean documents table
            logger.info("Cleaning documents table...")
            result = supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            logger.info(f"Deleted documents")
            
            logger.info("✓ Supabase cleanup complete!")
            
            # Verify cleanup
            self.verify_supabase_cleanup(supabase)
            
        except Exception as e:
            logger.error(f"Error cleaning Supabase: {e}")
            raise
    
    def verify_supabase_cleanup(self, supabase: Client):
        """Verify that all Supabase tables are empty."""
        try:
            logger.info("\nVerifying Supabase cleanup...")
            
            tables = ["documents", "chunks", "sessions", "messages"]
            all_empty = True
            
            for table in tables:
                result = supabase.table(table).select("id", count="exact").limit(1).execute()
                count = result.count if hasattr(result, 'count') else len(result.data)
                
                if count > 0:
                    logger.warning(f"  ✗ {table}: {count} records remaining")
                    all_empty = False
                else:
                    logger.info(f"  ✓ {table}: empty")
            
            if all_empty:
                logger.info("✓ All Supabase tables successfully cleaned!")
            else:
                logger.warning("⚠ Some tables still contain data")
                
        except Exception as e:
            logger.error(f"Error verifying Supabase cleanup: {e}")
    
    def clean_neo4j(self):
        """Clean all nodes and relationships from Neo4j."""
        try:
            logger.info("\nConnecting to Neo4j...")
            driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            with driver.session() as session:
                # Delete all nodes and relationships
                logger.info("Deleting all nodes and relationships...")
                result = session.run("""
                    MATCH (n)
                    DETACH DELETE n
                    RETURN COUNT(n) as deletedCount
                """)
                record = result.single()
                deleted_count = record["deletedCount"] if record else 0
                logger.info(f"Deleted {deleted_count} nodes and their relationships")
                
                # Verify cleanup
                self.verify_neo4j_cleanup(session)
                
            driver.close()
            logger.info("✓ Neo4j cleanup complete!")
            
        except Exception as e:
            logger.error(f"Error cleaning Neo4j: {e}")
            raise
    
    def verify_neo4j_cleanup(self, session):
        """Verify that Neo4j is completely empty."""
        try:
            logger.info("\nVerifying Neo4j cleanup...")
            
            # Count nodes
            result = session.run("MATCH (n) RETURN COUNT(n) as nodeCount")
            node_count = result.single()["nodeCount"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN COUNT(r) as relCount")
            rel_count = result.single()["relCount"]
            
            if node_count == 0 and rel_count == 0:
                logger.info(f"  ✓ Nodes: {node_count}")
                logger.info(f"  ✓ Relationships: {rel_count}")
                logger.info("✓ Neo4j successfully cleaned!")
            else:
                logger.warning(f"  ⚠ Nodes remaining: {node_count}")
                logger.warning(f"  ⚠ Relationships remaining: {rel_count}")
                
        except Exception as e:
            logger.error(f"Error verifying Neo4j cleanup: {e}")
    
    def run_cleanup(self):
        """Run the complete cleanup process."""
        logger.info("=" * 60)
        logger.info("Starting Complete Database Cleanup (FORCE MODE)")
        logger.info("=" * 60)
        
        # Clean Supabase
        try:
            self.clean_supabase()
        except Exception as e:
            logger.error(f"Failed to clean Supabase: {e}")
            return False
        
        # Clean Neo4j
        try:
            self.clean_neo4j()
        except Exception as e:
            logger.error(f"Failed to clean Neo4j: {e}")
            return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Complete database cleanup successful!")
        logger.info("Databases are ready for fresh ingestion.")
        logger.info("=" * 60)
        
        return True


def main():
    """Main execution function - force mode, no confirmation."""
    print("\n⚠️  FORCE MODE: Cleaning all data from both databases...")
    
    cleaner = DatabaseCleaner()
    success = cleaner.run_cleanup()
    
    if success:
        print("\n✅ Cleanup complete! You can now run fresh ingestion with:")
        print("   python -m ingestion.ingest --clean --verbose")
        sys.exit(0)
    else:
        print("\n❌ Cleanup encountered errors. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()