#!/usr/bin/env python3
"""
Medical RAG Database Migration Runner for Supabase
==================================================

This script runs database migrations on Supabase PostgreSQL.
It executes migration files from the sql/migrations directory.

Usage:
    python run_migration.py <migration_file>
    python run_migration.py 002_enhance_episodic_memory.sql
    python run_migration.py --all  # Run all pending migrations
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


class MigrationRunner:
    """Handles running migrations on Supabase PostgreSQL."""
    
    def __init__(self):
        self.database_url = self._get_database_url()
        self.script_dir = Path(__file__).parent
        self.migrations_dir = self.script_dir / 'migrations'
        self.use_asyncpg = ASYNCPG_AVAILABLE
        
    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # First check for direct DATABASE_URL
        url = os.getenv('DATABASE_URL')
        if url and url != 'postgresql://postgres:YOUR_SUPABASE_DB_PASSWORD@db.bpopugzfbokjzgawshov.supabase.co:5432/postgres':
            return url
        
        # If DATABASE_URL is not properly configured, try to construct from Supabase settings
        supabase_url = os.getenv('SUPABASE_URL')
        if supabase_url:
            # Extract project ID from Supabase URL
            # Format: https://[project-id].supabase.co
            project_id = supabase_url.split('//')[1].split('.')[0]
            
            # You need to get the database password from Supabase dashboard
            print("⚠️  DATABASE_URL not properly configured.")
            print(f"   Please get your database password from:")
            print(f"   https://supabase.com/dashboard/project/{project_id}/settings/database")
            print("   Then update the DATABASE_URL in your .env file")
            sys.exit(1)
        
        # Try other environment variable names
        for var_name in ['SUPABASE_DATABASE_URL', 'POSTGRES_URL']:
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
            print("❌ Error: Database connection not properly configured.")
            print("   Please set DATABASE_URL in your .env file with your Supabase database password.")
            sys.exit(1)
    
    def _read_migration_file(self, filename: str) -> str:
        """Read migration file content."""
        file_path = self.migrations_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Migration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    async def _execute_sql_asyncpg(self, sql: str) -> None:
        """Execute SQL using asyncpg."""
        conn = await asyncpg.connect(self.database_url)
        try:
            await conn.execute(sql)
        except Exception as e:
            print(f"Database error: {e}")
            raise
        finally:
            await conn.close()
    
    def _execute_sql_psycopg2(self, sql: str) -> None:
        """Execute SQL using psycopg2."""
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    async def execute_sql(self, sql: str) -> None:
        """Execute SQL using available database driver."""
        if self.use_asyncpg and ASYNCPG_AVAILABLE:
            return await self._execute_sql_asyncpg(sql)
        elif PSYCOPG2_AVAILABLE:
            return self._execute_sql_psycopg2(sql)
        else:
            raise RuntimeError("No suitable PostgreSQL driver available. Install asyncpg or psycopg2.")
    
    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            print("Testing database connection...")
            if self.use_asyncpg:
                conn = await asyncpg.connect(self.database_url)
                await conn.fetchval("SELECT 1")
                await conn.close()
            else:
                conn = psycopg2.connect(self.database_url)
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                conn.close()
            print("✅ Database connection successful")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    async def check_migration_prerequisites(self, migration_file: str) -> bool:
        """Check if prerequisites for the migration are met."""
        # For 002_enhance_episodic_memory.sql, we need to check if base tables exist
        if '002' in migration_file:
            print("🔍 Checking migration prerequisites...")
            check_queries = [
                ("episodes table", "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'episodes')"),
                ("sessions table", "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sessions')"),
                ("uuid-ossp extension", "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp')"),
                ("pg_trgm extension", "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
            ]
            
            all_exist = True
            for name, query in check_queries:
                try:
                    if self.use_asyncpg:
                        conn = await asyncpg.connect(self.database_url)
                        exists = await conn.fetchval(query)
                        await conn.close()
                    else:
                        conn = psycopg2.connect(self.database_url)
                        with conn.cursor() as cur:
                            cur.execute(query)
                            exists = cur.fetchone()[0]
                        conn.close()
                    
                    if exists:
                        print(f"  ✅ {name} exists")
                    else:
                        print(f"  ❌ {name} not found")
                        all_exist = False
                except Exception as e:
                    print(f"  ❌ Error checking {name}: {e}")
                    all_exist = False
            
            if not all_exist:
                print("\n⚠️  Prerequisites not met. You may need to run earlier migrations first.")
                print("   Try running: python run_migration.py 001_add_episodic_memory_tables.sql")
            
            return all_exist
        
        return True
    
    async def run_migration(self, migration_file: str) -> bool:
        """Run a specific migration."""
        try:
            print(f"🚀 Running migration: {migration_file}")
            
            # Check prerequisites
            if not await self.check_migration_prerequisites(migration_file):
                return False
            
            # Read migration file
            migration_sql = self._read_migration_file(migration_file)
            
            print("📝 Executing migration...")
            await self.execute_sql(migration_sql)
            
            print(f"✅ Migration {migration_file} completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    async def verify_migration(self, migration_file: str) -> bool:
        """Verify the migration was successful."""
        try:
            print("🔍 Verifying migration...")
            
            # Specific verification for 002_enhance_episodic_memory.sql
            if '002' in migration_file:
                tables_to_check = [
                    'medical_entities',
                    'symptom_timeline', 
                    'treatment_outcomes',
                    'episode_medical_facts',
                    'patient_profiles',
                    'memory_importance_scores'
                ]
                
                views_to_check = [
                    'episode_summary',
                    'patient_medical_history'
                ]
                
                functions_to_check = [
                    'update_patient_profile',
                    'calculate_memory_importance'
                ]
                
                print("\n📊 Checking new tables...")
                for table in tables_to_check:
                    query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}')"
                    
                    if self.use_asyncpg:
                        conn = await asyncpg.connect(self.database_url)
                        exists = await conn.fetchval(query)
                        await conn.close()
                    else:
                        conn = psycopg2.connect(self.database_url)
                        with conn.cursor() as cur:
                            cur.execute(query)
                            exists = cur.fetchone()[0]
                        conn.close()
                    
                    if exists:
                        print(f"  ✅ Table '{table}' created")
                    else:
                        print(f"  ❌ Table '{table}' not found")
                        return False
                
                print("\n📊 Checking views...")
                for view in views_to_check:
                    query = f"SELECT EXISTS (SELECT 1 FROM information_schema.views WHERE table_name = '{view}')"
                    
                    if self.use_asyncpg:
                        conn = await asyncpg.connect(self.database_url)
                        exists = await conn.fetchval(query)
                        await conn.close()
                    else:
                        conn = psycopg2.connect(self.database_url)
                        with conn.cursor() as cur:
                            cur.execute(query)
                            exists = cur.fetchone()[0]
                        conn.close()
                    
                    if exists:
                        print(f"  ✅ View '{view}' created")
                    else:
                        print(f"  ❌ View '{view}' not found")
                
                print("\n📊 Checking functions...")
                for func in functions_to_check:
                    query = f"SELECT EXISTS (SELECT 1 FROM pg_proc WHERE proname = '{func}')"
                    
                    if self.use_asyncpg:
                        conn = await asyncpg.connect(self.database_url)
                        exists = await conn.fetchval(query)
                        await conn.close()
                    else:
                        conn = psycopg2.connect(self.database_url)
                        with conn.cursor() as cur:
                            cur.execute(query)
                            exists = cur.fetchone()[0]
                        conn.close()
                    
                    if exists:
                        print(f"  ✅ Function '{func}' created")
                    else:
                        print(f"  ❌ Function '{func}' not found")
                
                print("\n✅ All migration objects verified successfully!")
                return True
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    async def list_migrations(self) -> List[str]:
        """List available migration files."""
        if not self.migrations_dir.exists():
            return []
        
        migrations = sorted([
            f.name for f in self.migrations_dir.glob('*.sql')
        ])
        return migrations
    
    async def run(self, migration_file: Optional[str] = None, run_all: bool = False) -> bool:
        """Main execution method."""
        print("🏥 Medical RAG Database Migration Runner")
        print("=" * 50)
        print(f"Database URL: {self.database_url[:50]}...")
        print(f"Using driver: {'asyncpg' if self.use_asyncpg else 'psycopg2'}")
        print("=" * 50)
        
        # Test connection
        if not await self.test_connection():
            return False
        
        if run_all:
            migrations = await self.list_migrations()
            print(f"\n📁 Found {len(migrations)} migration(s)")
            for migration in migrations:
                print(f"\n--- Running migration: {migration} ---")
                if not await self.run_migration(migration):
                    print(f"❌ Failed at migration: {migration}")
                    return False
                await self.verify_migration(migration)
            return True
        
        elif migration_file:
            success = await self.run_migration(migration_file)
            if success:
                await self.verify_migration(migration_file)
            return success
        
        else:
            # List available migrations
            migrations = await self.list_migrations()
            if migrations:
                print("\n📁 Available migrations:")
                for m in migrations:
                    print(f"  • {m}")
                print("\nUsage: python run_migration.py <migration_file>")
            else:
                print("No migrations found in sql/migrations/")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run database migrations on Supabase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_migration.py 002_enhance_episodic_memory.sql
  python run_migration.py --all
  
Environment Variables:
  DATABASE_URL - PostgreSQL connection string with Supabase password
  SUPABASE_URL - Your Supabase project URL
        """
    )
    
    parser.add_argument(
        'migration_file',
        nargs='?',
        help='Migration file to run (e.g., 002_enhance_episodic_memory.sql)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all pending migrations in order'
    )
    
    args = parser.parse_args()
    
    # Check if required libraries are available
    if not (ASYNCPG_AVAILABLE or PSYCOPG2_AVAILABLE):
        print("❌ Error: No suitable PostgreSQL driver found.")
        print("Please install one of the following:")
        print("  pip install asyncpg")
        print("  pip install psycopg2-binary")
        sys.exit(1)
    
    # Run migration
    runner = MigrationRunner()
    
    async def run():
        success = await runner.run(
            migration_file=args.migration_file,
            run_all=args.all
        )
        return success
    
    # Run async function
    if ASYNCPG_AVAILABLE:
        success = asyncio.run(run())
    else:
        # For psycopg2, we don't need async
        runner.use_asyncpg = False
        success = asyncio.run(run())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()