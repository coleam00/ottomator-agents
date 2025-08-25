#!/usr/bin/env python3
"""
Medical RAG Database Deployment Script for Supabase
===================================================

This script deploys the Medical RAG database schema to Supabase PostgreSQL.
It handles the complete setup including extensions, tables, functions, and verification.

Requirements:
- psycopg2 or asyncpg
- python-dotenv
- Environment variables for Supabase connection

Usage:
    python deploy_to_supabase.py
    python deploy_to_supabase.py --verify-only
    python deploy_to_supabase.py --clean-deploy
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Make sure environment variables are set manually.")


class SupabaseDeployment:
    """Handles deployment of Medical RAG schema to Supabase PostgreSQL."""
    
    def __init__(self):
        self.database_url = self._get_database_url()
        self.script_dir = Path(__file__).parent
        self.use_asyncpg = ASYNCPG_AVAILABLE
        
    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Try different environment variable names
        for var_name in ['DATABASE_URL', 'SUPABASE_DATABASE_URL', 'POSTGRES_URL']:
            url = os.getenv(var_name)
            if url:
                return url
                
        # Construct from individual components
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        user = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', '')
        database = os.getenv('DB_NAME', 'postgres')
        
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            return f"postgresql://{user}@{host}:{port}/{database}"
    
    def _read_sql_file(self, filename: str) -> str:
        """Read SQL file content."""
        file_path = self.script_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"SQL file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    async def _execute_sql_asyncpg(self, sql: str, fetch_results: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Execute SQL using asyncpg."""
        conn = await asyncpg.connect(self.database_url)
        try:
            if fetch_results:
                # Split SQL into individual statements for execution
                statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
                results = []
                
                for stmt in statements:
                    if stmt.upper().startswith('SELECT'):
                        rows = await conn.fetch(stmt)
                        results.extend([dict(row) for row in rows])
                    else:
                        await conn.execute(stmt)
                
                return results
            else:
                # Execute the entire SQL block
                await conn.execute(sql)
                return None
                
        except Exception as e:
            print(f"Database error: {e}")
            raise
        finally:
            await conn.close()
    
    def _execute_sql_psycopg2(self, sql: str, fetch_results: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Execute SQL using psycopg2."""
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if fetch_results:
                    # Split SQL into individual statements for execution
                    statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
                    results = []
                    
                    for stmt in statements:
                        if stmt.upper().startswith('SELECT'):
                            cur.execute(stmt)
                            rows = cur.fetchall()
                            results.extend([dict(row) for row in rows])
                        else:
                            cur.execute(stmt)
                    
                    conn.commit()
                    return results
                else:
                    # Execute the entire SQL block
                    cur.execute(sql)
                    conn.commit()
                    return None
                    
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    async def execute_sql(self, sql: str, fetch_results: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Execute SQL using available database driver."""
        if self.use_asyncpg and ASYNCPG_AVAILABLE:
            return await self._execute_sql_asyncpg(sql, fetch_results)
        elif PSYCOPG2_AVAILABLE:
            return self._execute_sql_psycopg2(sql, fetch_results)
        else:
            raise RuntimeError("No suitable PostgreSQL driver available. Install asyncpg or psycopg2.")
    
    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            print("Testing database connection...")
            await self.execute_sql("SELECT 1 as test;")
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    async def deploy_schema(self) -> bool:
        """Deploy the complete schema."""
        try:
            print("🚀 Starting Medical RAG schema deployment to Supabase...")
            
            # Read and execute the main schema
            schema_sql = self._read_sql_file('supabase_schema.sql')
            
            print("📝 Executing schema deployment...")
            await self.execute_sql(schema_sql)
            
            print("✅ Schema deployment completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Schema deployment failed: {e}")
            return False
    
    async def verify_deployment(self) -> bool:
        """Verify the deployment was successful."""
        try:
            print("🔍 Verifying deployment...")
            
            # Read and execute verification script
            verify_sql = self._read_sql_file('verify_setup.sql')
            results = await self.execute_sql(verify_sql, fetch_results=True)
            
            # Basic verification checks
            verification_queries = [
                "SELECT COUNT(*) as count FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm');",
                "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_name IN ('documents', 'chunks', 'sessions', 'messages');",
                "SELECT COUNT(*) as count FROM information_schema.routines WHERE routine_name IN ('match_chunks', 'hybrid_search', 'get_document_chunks');",
                "SELECT COUNT(*) as count FROM documents WHERE source = 'system';"
            ]
            
            checks_passed = 0
            total_checks = len(verification_queries)
            
            for i, query in enumerate(verification_queries):
                try:
                    result = await self.execute_sql(query, fetch_results=True)
                    if result and len(result) > 0:
                        count = result[0].get('count', 0)
                        expected_counts = [3, 4, 3, 1]  # Expected counts for each check
                        
                        if count >= expected_counts[i]:
                            print(f"✅ Check {i+1}/{total_checks}: Passed ({count} items found)")
                            checks_passed += 1
                        else:
                            print(f"❌ Check {i+1}/{total_checks}: Failed ({count} items found, expected {expected_counts[i]})")
                    else:
                        print(f"❌ Check {i+1}/{total_checks}: No results returned")
                except Exception as e:
                    print(f"❌ Check {i+1}/{total_checks}: Error - {e}")
            
            if checks_passed == total_checks:
                print(f"✅ All {total_checks} verification checks passed!")
                return True
            else:
                print(f"❌ {checks_passed}/{total_checks} verification checks passed")
                return False
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    async def get_database_stats(self) -> None:
        """Display database statistics."""
        try:
            print("📊 Database Statistics:")
            stats_sql = "SELECT * FROM get_database_stats();"
            results = await self.execute_sql(stats_sql, fetch_results=True)
            
            if results:
                stats = results[0]
                print(f"  • Documents: {stats.get('total_documents', 0)}")
                print(f"  • Chunks: {stats.get('total_chunks', 0)}")
                print(f"  • Sessions: {stats.get('total_sessions', 0)}")
                print(f"  • Messages: {stats.get('total_messages', 0)}")
                print(f"  • Chunks with embeddings: {stats.get('chunks_with_embeddings', 0)}")
                print(f"  • Average chunks per document: {stats.get('avg_chunks_per_document', 0)}")
                print(f"  • Database size: {stats.get('database_size', 'Unknown')}")
        
        except Exception as e:
            print(f"❌ Failed to get database stats: {e}")
    
    async def run_deployment(self, verify_only: bool = False, clean_deploy: bool = False) -> bool:
        """Run the complete deployment process."""
        print("🏥 Medical RAG Database Deployment for Supabase")
        print("=" * 50)
        print(f"Database URL: {self.database_url[:50]}...")
        print(f"Using driver: {'asyncpg' if self.use_asyncpg else 'psycopg2'}")
        print("=" * 50)
        
        # Test connection
        if not await self.test_connection():
            return False
        
        success = True
        
        if not verify_only:
            # Deploy schema
            if not await self.deploy_schema():
                success = False
        
        # Always run verification
        if not await self.verify_deployment():
            success = False
        
        # Show stats if deployment was successful
        if success:
            await self.get_database_stats()
            
            print("\n🎉 Deployment Summary:")
            print("✅ PostgreSQL extensions enabled (vector, uuid-ossp, pg_trgm)")
            print("✅ Core tables created (documents, chunks, sessions, messages)")
            print("✅ Indexes created for optimal performance")
            print("✅ Database functions created (match_chunks, hybrid_search, etc.)")
            print("✅ Triggers created for automatic timestamp updates")
            print("✅ Views created (document_summaries, session_summaries)")
            print("✅ Test data inserted and verified")
            print("\n📚 Your Medical RAG database is ready for use!")
            
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Medical RAG database schema to Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_to_supabase.py                    # Full deployment
  python deploy_to_supabase.py --verify-only      # Verification only
  python deploy_to_supabase.py --clean-deploy     # Clean deployment
  
Environment Variables:
  DATABASE_URL or SUPABASE_DATABASE_URL - PostgreSQL connection string
  
  Or individual components:
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
        """
    )
    
    parser.add_argument(
        '--verify-only', 
        action='store_true',
        help='Only run verification, skip schema deployment'
    )
    
    parser.add_argument(
        '--clean-deploy',
        action='store_true', 
        help='Perform a clean deployment (drops existing objects first)'
    )
    
    args = parser.parse_args()
    
    # Check if required libraries are available
    if not (ASYNCPG_AVAILABLE or PSYCOPG2_AVAILABLE):
        print("❌ Error: No suitable PostgreSQL driver found.")
        print("Please install one of the following:")
        print("  pip install asyncpg")
        print("  pip install psycopg2-binary")
        sys.exit(1)
    
    # Run deployment
    deployment = SupabaseDeployment()
    
    async def run():
        success = await deployment.run_deployment(
            verify_only=args.verify_only,
            clean_deploy=args.clean_deploy
        )
        return success
    
    # Run async function
    if ASYNCPG_AVAILABLE:
        success = asyncio.run(run())
    else:
        # For psycopg2, we don't need async
        deployment.use_asyncpg = False
        success = asyncio.run(run())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()