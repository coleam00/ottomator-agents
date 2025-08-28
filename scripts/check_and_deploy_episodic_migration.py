#!/usr/bin/env python3
"""
Check existing tables and prepare episodic memory migration for Supabase.
Since Supabase client doesn't support direct DDL execution, this script:
1. Checks what tables already exist
2. Provides instructions for manual migration execution
3. Verifies the migration after manual execution
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict

# Load environment variables
load_dotenv()


class EpisodicMemoryMigrationChecker:
    """Check and prepare episodic memory migration for Supabase."""
    
    # Tables that should exist after the migration
    EPISODIC_TABLES = [
        'episodes',
        'episode_references',
        'episode_relationships',
        'memory_summaries',
        'medical_entities',
        'symptom_timeline',
        'treatment_outcomes',
        'episode_medical_facts',
        'patient_profiles',
        'memory_importance_scores'
    ]
    
    # Core tables that should already exist
    CORE_TABLES = [
        'documents',
        'chunks',
        'sessions',
        'messages'
    ]
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.service_key:
            print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env file")
            sys.exit(1)
        
        # Extract project ID from URL
        self.project_id = self.supabase_url.split('.')[0].replace('https://', '')
        
        # Create Supabase client
        print(f"🔌 Connecting to Supabase project: {self.project_id}")
        self.client: Client = create_client(self.supabase_url, self.service_key)
        
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            # Try to query the table (limit 0 for just checking existence)
            self.client.table(table_name).select("*").limit(0).execute()
            return True
        except Exception as e:
            error_str = str(e)
            if 'relation' in error_str and 'does not exist' in error_str:
                return False
            elif 'Could not find the table' in error_str:
                return False
            else:
                # Some other error - assume table might exist
                print(f"⚠️  Warning: Could not definitively check table '{table_name}': {error_str}")
                return None
    
    def check_existing_tables(self) -> Dict[str, List[str]]:
        """Check which tables already exist."""
        print("\n🔍 Checking existing database tables...")
        print("=" * 60)
        
        results = {
            'core_exists': [],
            'core_missing': [],
            'episodic_exists': [],
            'episodic_missing': [],
            'uncertain': []
        }
        
        # Check core tables
        print("\n📊 Core Tables (should already exist):")
        for table in self.CORE_TABLES:
            exists = self.check_table_exists(table)
            if exists is True:
                print(f"  ✅ {table:<25} - EXISTS")
                results['core_exists'].append(table)
            elif exists is False:
                print(f"  ❌ {table:<25} - MISSING")
                results['core_missing'].append(table)
            else:
                print(f"  ⚠️  {table:<25} - UNCERTAIN")
                results['uncertain'].append(table)
        
        # Check episodic memory tables
        print("\n🧠 Episodic Memory Tables (to be created):")
        for table in self.EPISODIC_TABLES:
            exists = self.check_table_exists(table)
            if exists is True:
                print(f"  ✅ {table:<25} - Already EXISTS")
                results['episodic_exists'].append(table)
            elif exists is False:
                print(f"  ⏳ {table:<25} - To be created")
                results['episodic_missing'].append(table)
            else:
                print(f"  ⚠️  {table:<25} - UNCERTAIN")
                results['uncertain'].append(table)
        
        return results
    
    def prepare_migration_file(self) -> str:
        """Prepare the migration file and return its path."""
        migration_file = Path(__file__).parent.parent / "sql" / "combined_episodic_memory_migration.sql"
        
        if not migration_file.exists():
            print(f"\n❌ Migration file not found: {migration_file}")
            return None
        
        print(f"\n📄 Migration file ready: {migration_file}")
        
        # Also copy to temp for easy access
        temp_file = Path("/tmp/episodic_memory_migration.sql")
        try:
            with open(migration_file, 'r') as source:
                content = source.read()
            with open(temp_file, 'w') as dest:
                dest.write(content)
            print(f"📋 Also copied to: {temp_file}")
            return str(temp_file)
        except Exception as e:
            print(f"⚠️  Could not create temp copy: {e}")
            return str(migration_file)
    
    def print_migration_instructions(self, results: Dict[str, List[str]]):
        """Print instructions for running the migration manually."""
        print("\n" + "=" * 60)
        print("📚 MIGRATION INSTRUCTIONS")
        print("=" * 60)
        
        # Check if prerequisites are met
        if results['core_missing']:
            print("\n⚠️  WARNING: Core tables are missing!")
            print("   The following core tables should exist first:")
            for table in results['core_missing']:
                print(f"   - {table}")
            print("\n   Please ensure the base schema is deployed before running this migration.")
            print("   Run: python sql/scripts/deploy_schema_via_api.py")
            return False
        
        if results['episodic_exists']:
            print(f"\n✅ {len(results['episodic_exists'])} episodic tables already exist")
            if len(results['episodic_exists']) == len(self.EPISODIC_TABLES):
                print("   All episodic memory tables are already created!")
                print("   Migration may have already been run.")
                return True
        
        if results['episodic_missing']:
            print(f"\n📦 {len(results['episodic_missing'])} tables will be created by the migration")
        
        # Print manual execution instructions
        print("\n🚀 TO RUN THE MIGRATION:")
        print("-" * 40)
        
        dashboard_url = f"https://supabase.com/dashboard/project/{self.project_id}/sql/new"
        print(f"\n1️⃣  Open Supabase SQL Editor:")
        print(f"   {dashboard_url}")
        
        print(f"\n2️⃣  Copy the migration SQL from:")
        print(f"   sql/combined_episodic_memory_migration.sql")
        
        print(f"\n3️⃣  Paste the entire SQL content into the editor")
        
        print(f"\n4️⃣  Click 'Run' button (or press Cmd/Ctrl + Enter)")
        
        print(f"\n5️⃣  Wait for completion (should take < 30 seconds)")
        
        print("\n" + "-" * 40)
        print("ALTERNATIVE: Using psql command line")
        print("-" * 40)
        
        print("\n1️⃣  Get your database password from:")
        print(f"   https://supabase.com/dashboard/project/{self.project_id}/settings/database")
        
        print("\n2️⃣  Set the DATABASE_URL in your .env file:")
        print("   DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.{}.supabase.co:5432/postgres".format(self.project_id))
        
        print("\n3️⃣  Run the migration:")
        print("   psql -d \"$DATABASE_URL\" -f sql/combined_episodic_memory_migration.sql")
        
        return True
    
    def verify_migration(self) -> bool:
        """Verify the migration was successful after manual execution."""
        print("\n" + "=" * 60)
        print("🔍 POST-MIGRATION VERIFICATION")
        print("=" * 60)
        
        print("\nAfter running the migration, re-run this script to verify:")
        print("  python check_and_deploy_episodic_migration.py")
        
        # Count existing episodic tables
        existing_count = 0
        for table in self.EPISODIC_TABLES:
            if self.check_table_exists(table):
                existing_count += 1
        
        if existing_count == len(self.EPISODIC_TABLES):
            print(f"\n✅ SUCCESS: All {len(self.EPISODIC_TABLES)} episodic memory tables exist!")
            return True
        elif existing_count > 0:
            print(f"\n⚠️  PARTIAL: {existing_count}/{len(self.EPISODIC_TABLES)} tables exist")
            print("   The migration may be partially complete.")
            return False
        else:
            print(f"\n❌ NOT RUN: No episodic memory tables found")
            print("   Please run the migration using the instructions above.")
            return False
    
    def run(self):
        """Main execution flow."""
        print("\n🧠 Episodic Memory Migration Checker for Supabase")
        print("=" * 60)
        print(f"📍 Project: {self.project_id}")
        print(f"🌐 URL: {self.supabase_url}")
        print("=" * 60)
        
        # Check existing tables
        results = self.check_existing_tables()
        
        # Prepare migration file
        migration_path = self.prepare_migration_file()
        
        if not migration_path:
            return False
        
        # Print instructions
        can_proceed = self.print_migration_instructions(results)
        
        if not can_proceed:
            return False
        
        # Verification info
        if results['episodic_missing']:
            self.verify_migration()
        
        return True


def main():
    """Main entry point."""
    checker = EpisodicMemoryMigrationChecker()
    success = checker.run()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Migration check complete!")
    else:
        print("❌ Migration check failed - please address the issues above")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()