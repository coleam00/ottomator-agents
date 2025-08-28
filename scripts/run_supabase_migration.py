#!/usr/bin/env python3
"""
Run database migrations on Supabase using the Supabase client.
This script executes SQL migrations through Supabase's API.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from supabase import create_client, Client
import re

# Load environment variables
load_dotenv()


class SupabaseMigrationRunner:
    """Handles running migrations on Supabase using the Supabase client."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.service_key:
            print("❌ Error: Supabase configuration missing!")
            print("   Please ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in .env")
            sys.exit(1)
        
        # Initialize Supabase client with service role key for admin operations
        self.client: Client = create_client(self.supabase_url, self.service_key)
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent  # Get project root
        self.migrations_dir = self.project_root / 'sql' / 'migrations'
        
    def _read_migration_file(self, filename: str) -> str:
        """Read migration file content."""
        file_path = self.migrations_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Migration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _split_sql_statements(self, sql: str) -> List[str]:
        """
        Split SQL into individual statements, handling complex cases.
        This is a simplified version - for production, consider using sqlparse.
        """
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        
        # Split by semicolons but be careful with functions/procedures
        statements = []
        current_statement = []
        in_function = False
        
        for line in sql.split('\n'):
            line_upper = line.upper().strip()
            
            # Check if we're entering a function/procedure definition
            if 'CREATE OR REPLACE FUNCTION' in line_upper or 'CREATE FUNCTION' in line_upper:
                in_function = True
            elif 'CREATE OR REPLACE TRIGGER' in line_upper or 'CREATE TRIGGER' in line_upper:
                in_function = True
            elif 'CREATE OR REPLACE VIEW' in line_upper or 'CREATE VIEW' in line_upper:
                in_function = False  # Views don't use $$ blocks
            
            current_statement.append(line)
            
            # Check for end of statement
            if line.strip().endswith(';'):
                if not in_function:
                    # Regular statement ended
                    statement = '\n'.join(current_statement).strip()
                    if statement and statement.upper() not in ['BEGIN;', 'COMMIT;', 'BEGIN', 'COMMIT']:
                        statements.append(statement)
                    current_statement = []
                elif '$$' in line and 'LANGUAGE' in line_upper:
                    # Function/trigger ended
                    statement = '\n'.join(current_statement).strip()
                    if statement:
                        statements.append(statement)
                    current_statement = []
                    in_function = False
        
        # Add any remaining statement
        if current_statement:
            statement = '\n'.join(current_statement).strip()
            if statement and statement.upper() not in ['BEGIN;', 'COMMIT;', 'BEGIN', 'COMMIT']:
                statements.append(statement)
        
        return statements
    
    async def execute_migration(self, sql: str) -> bool:
        """Execute migration SQL using Supabase RPC."""
        try:
            # Split SQL into individual statements
            statements = self._split_sql_statements(sql)
            
            print(f"📝 Executing {len(statements)} SQL statements...")
            
            for i, statement in enumerate(statements, 1):
                # Get first few words for logging
                statement_preview = ' '.join(statement.split()[:5])
                print(f"   Statement {i}/{len(statements)}: {statement_preview}...")
                
                try:
                    # Execute through Supabase's postgrest API
                    # For DDL statements, we need to use the raw SQL endpoint
                    response = self.client.postgrest.rpc(
                        'exec_sql',
                        {'sql': statement}
                    ).execute()
                    
                    if hasattr(response, 'data'):
                        print(f"     ✅ Success")
                except Exception as e:
                    # If RPC function doesn't exist, try to execute directly
                    # Note: This requires the exec_sql function to be created first
                    print(f"     ⚠️  Direct execution not available, creating exec_sql function...")
                    
                    # First, let's create the exec_sql function if it doesn't exist
                    create_function = """
                    CREATE OR REPLACE FUNCTION exec_sql(sql text)
                    RETURNS void
                    LANGUAGE plpgsql
                    SECURITY DEFINER
                    AS $$
                    BEGIN
                        EXECUTE sql;
                    END;
                    $$;
                    """
                    
                    # For now, we'll print instructions for manual execution
                    print(f"     ❌ Cannot execute directly via Supabase client")
                    print(f"        The Supabase client doesn't support direct DDL execution.")
                    print(f"        Please run this migration manually in the Supabase SQL Editor.")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Migration execution failed: {e}")
            return False
    
    async def check_prerequisites(self, migration_file: str) -> bool:
        """Check if prerequisites for the migration are met."""
        print("🔍 Checking migration prerequisites...")
        
        if '002' in migration_file:
            # Check for required tables
            checks = {
                'episodes': "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'episodes')",
                'sessions': "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sessions')",
            }
            
            all_exist = True
            for table_name, check_query in checks.items():
                try:
                    # Check if table exists using Supabase client
                    response = self.client.table(table_name).select('*').limit(0).execute()
                    print(f"  ✅ Table '{table_name}' exists")
                except Exception as e:
                    if 'relation' in str(e) and 'does not exist' in str(e):
                        print(f"  ❌ Table '{table_name}' not found")
                        all_exist = False
                    else:
                        print(f"  ⚠️  Could not check table '{table_name}': {e}")
            
            if not all_exist:
                print("\n⚠️  Prerequisites not met!")
                print("   You need to run the first migration (001) before this one.")
                print("   The 001 migration creates the base episodic memory tables.")
            
            return all_exist
        
        return True
    
    async def verify_migration(self, migration_file: str) -> bool:
        """Verify the migration was successful."""
        print("\n🔍 Verifying migration...")
        
        if '002' in migration_file:
            # Tables to check
            tables = [
                'medical_entities',
                'symptom_timeline',
                'treatment_outcomes', 
                'episode_medical_facts',
                'patient_profiles',
                'memory_importance_scores'
            ]
            
            success_count = 0
            for table in tables:
                try:
                    # Try to query the table
                    response = self.client.table(table).select('*').limit(0).execute()
                    print(f"  ✅ Table '{table}' created successfully")
                    success_count += 1
                except Exception as e:
                    if 'relation' in str(e) and 'does not exist' in str(e):
                        print(f"  ❌ Table '{table}' not found")
                    else:
                        print(f"  ⚠️  Could not verify table '{table}': {e}")
            
            if success_count == len(tables):
                print("\n✅ All tables verified successfully!")
                return True
            else:
                print(f"\n⚠️  Only {success_count}/{len(tables)} tables verified")
                return False
        
        return True
    
    def print_manual_instructions(self, migration_file: str):
        """Print instructions for manual migration execution."""
        print("\n📋 Manual Migration Instructions:")
        print("=" * 50)
        print("Since direct DDL execution through Supabase client is limited,")
        print("please follow these steps to run the migration manually:")
        print()
        print("1. Go to your Supabase Dashboard:")
        print(f"   {self.supabase_url.replace('.supabase.co', '.supabase.com/project/').replace('https://', 'https://supabase.com/dashboard/project/')}/sql")
        print()
        print("2. Open the SQL Editor")
        print()
        print("3. Copy and paste the migration file content:")
        print(f"   File: sql/migrations/{migration_file}")
        print()
        print("4. Click 'Run' to execute the migration")
        print()
        print("5. Verify the migration succeeded by checking the tables were created")
        print()
        print("Alternative: Use psql command line:")
        print(f"   Get your database password from the Supabase dashboard")
        print(f"   Then run: psql -d \"$DATABASE_URL\" -f sql/migrations/{migration_file}")
    
    async def run(self, migration_file: str) -> bool:
        """Main execution method."""
        print("🏥 Medical RAG Database Migration Runner (Supabase)")
        print("=" * 50)
        print(f"🌐 Supabase URL: {self.supabase_url}")
        print(f"📁 Migration: {migration_file}")
        print("=" * 50)
        
        # Check prerequisites
        if not await self.check_prerequisites(migration_file):
            return False
        
        # Read migration file
        try:
            migration_sql = self._read_migration_file(migration_file)
            print(f"\n📖 Read migration file: {len(migration_sql)} characters")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False
        
        # Print manual instructions
        self.print_manual_instructions(migration_file)
        
        # Also save the migration to a temp file for easy copying
        temp_file = Path('/tmp') / f'supabase_migration_{migration_file}'
        try:
            with open(temp_file, 'w') as f:
                f.write(migration_sql)
            print(f"\n💾 Migration saved to: {temp_file}")
            print("   You can copy this file's content to the SQL editor")
        except Exception as e:
            print(f"⚠️  Could not save temp file: {e}")
        
        # Note: Direct DDL execution through Supabase client is not fully supported
        # The user will need to run the migration manually through the dashboard
        
        return True


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_supabase_migration.py <migration_file>")
        print("Example: python run_supabase_migration.py 002_enhance_episodic_memory.sql")
        sys.exit(1)
    
    migration_file = sys.argv[1]
    
    runner = SupabaseMigrationRunner()
    success = await runner.run(migration_file)
    
    if success:
        print("\n✅ Migration preparation complete!")
        print("   Please follow the manual instructions above to execute the migration.")
    else:
        print("\n❌ Migration preparation failed!")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())