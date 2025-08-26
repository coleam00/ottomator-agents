#!/usr/bin/env python3
"""
Script to run the episodic memory migration on Supabase database.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def run_migration():
    """Run the episodic memory migration on Supabase."""
    
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env file")
        sys.exit(1)
    
    # Create Supabase client
    print(f"Connecting to Supabase at {supabase_url}...")
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Read the migration file
    migration_file = Path("sql/combined_episodic_memory_migration.sql")
    if not migration_file.exists():
        print(f"Error: Migration file not found at {migration_file}")
        sys.exit(1)
    
    print(f"Reading migration file from {migration_file}...")
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    # Split the migration into individual statements
    # Remove comments and split by semicolons
    statements = []
    current_statement = []
    in_function = False
    
    for line in migration_sql.split('\n'):
        # Skip pure comment lines
        if line.strip().startswith('--') and not in_function:
            continue
        
        # Check if we're entering or leaving a function/trigger definition
        if '$$' in line:
            in_function = not in_function
        
        current_statement.append(line)
        
        # If we hit a semicolon and we're not in a function, this statement is complete
        if line.strip().endswith(';') and not in_function:
            statement = '\n'.join(current_statement).strip()
            if statement and not statement.startswith('--'):
                statements.append(statement)
            current_statement = []
    
    # Add any remaining statement
    if current_statement:
        statement = '\n'.join(current_statement).strip()
        if statement and not statement.startswith('--'):
            statements.append(statement)
    
    print(f"Found {len(statements)} SQL statements to execute")
    
    # Execute each statement
    successful = 0
    failed = 0
    errors = []
    
    for i, statement in enumerate(statements, 1):
        # Skip the verification query at the end
        if 'information_schema.tables' in statement or 'information_schema.views' in statement:
            continue
            
        try:
            # Get the first few words of the statement for logging
            statement_preview = ' '.join(statement.split()[:5])
            if len(statement_preview) > 50:
                statement_preview = statement_preview[:47] + "..."
            
            print(f"[{i}/{len(statements)}] Executing: {statement_preview}")
            
            # Execute the statement using Supabase RPC
            # We'll use the raw SQL execution through Supabase's postgres functions
            result = supabase.rpc('exec_sql', {'query': statement}).execute()
            successful += 1
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a "already exists" error which we can ignore
            if 'already exists' in error_msg.lower():
                print(f"  ⚠️  Already exists (skipping): {statement_preview}")
                successful += 1
            else:
                print(f"  ❌ Failed: {error_msg[:100]}")
                failed += 1
                errors.append(f"Statement {i}: {statement_preview} - Error: {error_msg[:200]}")
    
    print("\n" + "="*60)
    print("Migration Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Verify tables were created
    print("\n" + "="*60)
    print("Verifying migration...")
    
    try:
        # Check for tables
        table_check = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN (
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
        )
        ORDER BY table_name;
        """
        
        result = supabase.rpc('exec_sql', {'query': table_check}).execute()
        
        if result.data:
            print(f"\n✅ Found {len(result.data)} tables:")
            for table in result.data:
                print(f"  - {table['table_name']}")
        else:
            print("\n⚠️  No episodic memory tables found")
            
    except Exception as e:
        print(f"\n❌ Could not verify tables: {e}")
    
    print("\nMigration complete!")

if __name__ == "__main__":
    run_migration()