#!/usr/bin/env python3
"""
Verify that the episodic memory migration was successfully applied to Supabase.
"""

import os
import sys
import asyncio
from typing import Dict, List
from dotenv import load_dotenv
from supabase import create_client, Client
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()


class MigrationVerifier:
    """Verifies the episodic memory migration on Supabase."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url or not self.service_key:
            console.print("[red]❌ Error: Supabase configuration missing![/red]")
            console.print("   Please ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in .env")
            sys.exit(1)
        
        # Initialize Supabase client
        self.client: Client = create_client(self.supabase_url, self.service_key)
    
    async def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists by attempting to query it."""
        try:
            # Try to query the table (limit 0 for no data)
            response = self.client.table(table_name).select('*').limit(0).execute()
            return True
        except Exception as e:
            if 'relation' in str(e) and 'does not exist' in str(e):
                return False
            # If it's a different error, the table might exist but have other issues
            console.print(f"[yellow]⚠️  Could not verify table '{table_name}': {e}[/yellow]")
            return None
    
    async def verify_migration_001(self) -> Dict[str, bool]:
        """Verify tables from migration 001."""
        tables = [
            'episodes',
            'episode_references',
            'episode_relationships',
            'memory_summaries'
        ]
        
        results = {}
        for table in tables:
            exists = await self.check_table_exists(table)
            results[table] = exists
        
        return results
    
    async def verify_migration_002(self) -> Dict[str, bool]:
        """Verify tables from migration 002."""
        tables = [
            'medical_entities',
            'symptom_timeline',
            'treatment_outcomes',
            'episode_medical_facts',
            'patient_profiles',
            'memory_importance_scores'
        ]
        
        results = {}
        for table in tables:
            exists = await self.check_table_exists(table)
            results[table] = exists
        
        return results
    
    async def verify_views(self) -> Dict[str, bool]:
        """Verify views were created."""
        views = [
            'episode_summary',
            'patient_medical_history'
        ]
        
        results = {}
        for view in views:
            exists = await self.check_table_exists(view)  # Views can be queried like tables
            results[view] = exists
        
        return results
    
    async def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        all_tables = [
            'episodes', 'episode_references', 'episode_relationships', 'memory_summaries',
            'medical_entities', 'symptom_timeline', 'treatment_outcomes', 
            'episode_medical_facts', 'patient_profiles', 'memory_importance_scores'
        ]
        
        counts = {}
        for table in all_tables:
            try:
                response = self.client.table(table).select('id', count='exact').execute()
                counts[table] = response.count if hasattr(response, 'count') else 0
            except Exception:
                counts[table] = -1  # -1 indicates table doesn't exist or error
        
        return counts
    
    def print_results(self, migration_001: Dict, migration_002: Dict, views: Dict, counts: Dict):
        """Print verification results in a nice format."""
        console.print("\n[bold cyan]🔍 Episodic Memory Migration Verification[/bold cyan]")
        console.print("=" * 60)
        
        # Migration 001 Tables
        table = Table(title="Migration 001: Base Episodic Memory Tables")
        table.add_column("Table", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Row Count", style="yellow")
        
        for table_name, exists in migration_001.items():
            status = "✅ Exists" if exists else "❌ Missing"
            count = counts.get(table_name, -1)
            count_str = str(count) if count >= 0 else "N/A"
            table.add_row(table_name, status, count_str)
        
        console.print(table)
        console.print()
        
        # Migration 002 Tables
        table2 = Table(title="Migration 002: Medical Entity Tables")
        table2.add_column("Table", style="cyan")
        table2.add_column("Status", style="green")
        table2.add_column("Row Count", style="yellow")
        
        for table_name, exists in migration_002.items():
            status = "✅ Exists" if exists else "❌ Missing"
            count = counts.get(table_name, -1)
            count_str = str(count) if count >= 0 else "N/A"
            table2.add_row(table_name, status, count_str)
        
        console.print(table2)
        console.print()
        
        # Views
        table3 = Table(title="Database Views")
        table3.add_column("View", style="cyan")
        table3.add_column("Status", style="green")
        
        for view_name, exists in views.items():
            status = "✅ Exists" if exists else "❌ Missing"
            table3.add_row(view_name, status)
        
        console.print(table3)
        console.print()
        
        # Summary
        all_tables_001 = all(migration_001.values())
        all_tables_002 = all(migration_002.values())
        all_views = all(views.values())
        
        if all_tables_001 and all_tables_002 and all_views:
            console.print(Panel.fit(
                "[bold green]✅ All migrations successfully verified![/bold green]\n"
                "The episodic memory system is ready for use.",
                title="Success",
                border_style="green"
            ))
        elif all_tables_001 and not all_tables_002:
            console.print(Panel.fit(
                "[bold yellow]⚠️  Migration 001 complete, but Migration 002 is missing![/bold yellow]\n"
                "Run Migration 002 to add medical entity tracking.",
                title="Partial Success",
                border_style="yellow"
            ))
        elif not all_tables_001:
            console.print(Panel.fit(
                "[bold red]❌ Migration 001 is not complete![/bold red]\n"
                "You need to run Migration 001 first before Migration 002.",
                title="Migration Required",
                border_style="red"
            ))
        else:
            console.print(Panel.fit(
                "[bold yellow]⚠️  Some components are missing![/bold yellow]\n"
                "Please check the tables above and run the missing migrations.",
                title="Incomplete Migration",
                border_style="yellow"
            ))
    
    async def run(self):
        """Run the verification process."""
        console.print("[bold]🏥 Medical RAG Episodic Memory Migration Verifier[/bold]")
        console.print(f"🌐 Supabase URL: {self.supabase_url}")
        console.print()
        
        # Verify migrations
        migration_001 = await self.verify_migration_001()
        migration_002 = await self.verify_migration_002()
        views = await self.verify_views()
        counts = await self.get_table_counts()
        
        # Print results
        self.print_results(migration_001, migration_002, views, counts)
        
        # Return success status
        return all(migration_001.values()) and all(migration_002.values()) and all(views.values())


async def main():
    """Main entry point."""
    verifier = MigrationVerifier()
    success = await verifier.run()
    
    if success:
        console.print("\n[bold cyan]📝 Next Steps:[/bold cyan]")
        console.print("1. The episodic memory system is ready to use")
        console.print("2. Episodes will be automatically created during conversations")
        console.print("3. Medical entities will be extracted and tracked")
        console.print("4. Run the API server: python -m agent.api")
        console.print("5. Test with: python cli.py")
    else:
        console.print("\n[bold yellow]📝 To Run Migrations:[/bold yellow]")
        console.print("1. Go to your Supabase Dashboard SQL Editor:")
        console.print(f"   https://supabase.com/dashboard/project/{verifier.supabase_url.split('//')[1].split('.')[0]}/sql")
        console.print("2. Copy the content from: sql/combined_episodic_memory_migration.sql")
        console.print("3. Paste and run in the SQL Editor")
        console.print("4. Run this verification script again to confirm")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Check if rich is installed
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
    except ImportError:
        print("Installing rich for better output formatting...")
        os.system("pip install rich")
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
    
    asyncio.run(main())