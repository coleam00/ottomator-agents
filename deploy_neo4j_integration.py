#!/usr/bin/env python3
"""
Deploy Neo4j User Integration to Supabase
This script helps deploy the database migration and Edge Function
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

class Neo4jIntegrationDeployer:
    """Deploy Neo4j integration to Supabase"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8058")
        
        if not self.supabase_url or not self.service_key:
            print("❌ Error: Supabase configuration missing!")
            print("   Please ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in .env")
            sys.exit(1)
        
        self.client: Client = create_client(self.supabase_url, self.service_key)
        self.project_ref = self.supabase_url.split('//')[1].split('.')[0]
        self.script_dir = Path(__file__).parent
        
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "=" * 60)
        print(f"🚀 {title}")
        print("=" * 60)
        
    def deploy_database_migration(self):
        """Deploy database migration"""
        self.print_header("Database Migration Deployment")
        
        migration_file = self.script_dir / 'sql' / 'migrations' / '003_neo4j_user_integration.sql'
        
        if not migration_file.exists():
            print(f"❌ Migration file not found: {migration_file}")
            return False
        
        print(f"📁 Migration file: {migration_file}")
        print(f"📏 File size: {migration_file.stat().st_size} bytes")
        
        # Read migration content
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        print("\n📋 MANUAL DEPLOYMENT INSTRUCTIONS:")
        print("=" * 50)
        print(f"1. Open Supabase SQL Editor:")
        print(f"   https://supabase.com/dashboard/project/{self.project_ref}/sql")
        print("\n2. Copy and paste the migration from:")
        print(f"   {migration_file}")
        print("\n3. Click 'Run' to execute the migration")
        print("\n4. Verify the following were created:")
        print("   - Table: neo4j_users")
        print("   - Function: register_user_in_neo4j()")
        print("   - Function: update_neo4j_registration_status()")
        print("   - Function: get_pending_neo4j_registrations()")
        print("   - Trigger: trigger_register_neo4j_user")
        
        # Save to temp file for easy copying
        temp_file = Path('/tmp') / f'neo4j_migration_{self.project_ref}.sql'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(migration_sql)
        
        print(f"\n💾 Migration saved to: {temp_file}")
        print("   You can copy this file's content to the SQL editor")
        
        return True
    
    def deploy_edge_function(self):
        """Deploy Edge Function"""
        self.print_header("Edge Function Deployment")
        
        function_name = "register-neo4j-user"
        function_dir = self.script_dir / 'supabase' / 'functions' / function_name
        
        if not function_dir.exists():
            print(f"❌ Function directory not found: {function_dir}")
            return False
        
        print(f"📁 Function: {function_name}")
        print(f"📂 Location: {function_dir}")
        
        # Check if Supabase CLI is available
        try:
            result = subprocess.run(['supabase', '--version'], capture_output=True, text=True)
            print(f"✅ Supabase CLI version: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Supabase CLI not found. Please install it first:")
            print("   npm install -g supabase")
            return False
        
        print("\n📋 EDGE FUNCTION DEPLOYMENT:")
        print("=" * 50)
        
        # Generate deployment commands
        commands = [
            f"# 1. Link to your Supabase project (if not already linked)",
            f"supabase link --project-ref {self.project_ref}",
            f"",
            f"# 2. Set environment variables for the Edge Function",
            f"supabase secrets set BACKEND_API_URL={self.backend_url}",
            f"",
            f"# 3. Deploy the Edge Function",
            f"supabase functions deploy {function_name}",
            f"",
            f"# 4. Test the Edge Function (optional)",
            f"supabase functions invoke {function_name} --body '{{\"user_id\":\"test-user-id\"}}'",
        ]
        
        for cmd in commands:
            print(cmd)
        
        # Save deployment script
        deploy_script = Path('/tmp') / f'deploy_edge_function_{self.project_ref}.sh'
        with open(deploy_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Deploy Edge Function for {self.project_ref}\n\n")
            f.write("\n".join(commands))
        
        os.chmod(deploy_script, 0o755)
        
        print(f"\n💾 Deployment script saved to: {deploy_script}")
        print("   You can run: bash " + str(deploy_script))
        
        return True
    
    def verify_deployment(self):
        """Verify the deployment"""
        self.print_header("Deployment Verification")
        
        print("🔍 Checking database objects...")
        
        # Check if neo4j_users table exists
        try:
            response = self.client.table('neo4j_users').select('*').limit(1).execute()
            print("✅ Table 'neo4j_users' exists")
            
            # Check for any pending registrations
            if hasattr(response, 'data'):
                count = len(response.data) if response.data else 0
                print(f"   Found {count} registration records")
                
        except Exception as e:
            if 'relation' in str(e) and 'does not exist' in str(e):
                print("❌ Table 'neo4j_users' not found - migration needs to be run")
            else:
                print(f"⚠️  Could not verify table: {e}")
        
        print("\n📋 MANUAL VERIFICATION STEPS:")
        print("=" * 50)
        print("1. In Supabase SQL Editor, run:")
        print("   SELECT * FROM neo4j_users;")
        print("   SELECT * FROM pg_trigger WHERE tgname = 'trigger_register_neo4j_user';")
        print("\n2. Test Edge Function:")
        print(f"   curl -X POST https://{self.project_ref}.supabase.co/functions/v1/register-neo4j-user \\")
        print("     -H 'Authorization: Bearer YOUR_ANON_KEY' \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"user_id\": \"test-user-id\"}'")
        
        return True
    
    def create_test_user(self):
        """Create a test user to trigger the integration"""
        self.print_header("Test User Creation")
        
        print("📋 TO CREATE A TEST USER:")
        print("=" * 50)
        print("In Supabase Auth Dashboard or using the client SDK:")
        print("\n1. Create a new user with email/password")
        print("2. Check the neo4j_users table for a pending registration")
        print("3. Manually invoke the Edge Function to process registration")
        print("4. Verify the registration status updates to 'registered'")
        
        return True
    
    def run(self):
        """Run the deployment process"""
        print("\n" + "🔧" * 30)
        print("Neo4j User Integration Deployment")
        print("🔧" * 30)
        
        print(f"\n🌐 Supabase Project: {self.project_ref}")
        print(f"🔗 Backend API URL: {self.backend_url}")
        
        steps = [
            ("Database Migration", self.deploy_database_migration),
            ("Edge Function", self.deploy_edge_function),
            ("Verification", self.verify_deployment),
            ("Test Setup", self.create_test_user),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ Deployment step failed: {step_name}")
                print("   Please follow the manual instructions above")
        
        print("\n" + "=" * 60)
        print("✨ DEPLOYMENT PREPARATION COMPLETE!")
        print("=" * 60)
        print("\nIMPORTANT NEXT STEPS:")
        print("1. Execute the database migration in Supabase SQL Editor")
        print("2. Deploy the Edge Function using Supabase CLI")
        print("3. Configure BACKEND_API_URL environment variable")
        print("4. Test the integration with a new user registration")
        print("\n📚 Documentation:")
        print("   - Migration: sql/migrations/003_neo4j_user_integration.sql")
        print("   - Edge Function: supabase/functions/register-neo4j-user/")

if __name__ == "__main__":
    deployer = Neo4jIntegrationDeployer()
    deployer.run()