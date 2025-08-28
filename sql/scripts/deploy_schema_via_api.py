"""Deploy database schema to Supabase using the API"""
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Get Supabase credentials
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
    print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    exit(1)

# Create Supabase client
supabase: Client = create_client(url, key)

# Read the deployment SQL
with open("deploy_supabase_manual.sql", "r") as f:
    sql = f.read()

print("🚀 Deploying database schema to Supabase...")
print(f"   Project: {url}")
print("=" * 50)

try:
    # Execute the SQL via Supabase RPC
    # Note: This requires the sql_admin function to be available
    # Alternative: Use the Supabase dashboard SQL editor
    
    # Split SQL into individual statements
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    
    print(f"📝 Found {len(statements)} SQL statements to execute")
    
    # For now, just show instructions since direct SQL execution requires admin access
    print("\n⚠️  Direct SQL execution via API requires admin access.")
    print("\n✅ RECOMMENDED: Deploy manually via Supabase Dashboard:")
    print(f"   1. Go to: {url.replace('https://', 'https://supabase.com/dashboard/project/').replace('.supabase.co', '')}/sql")
    print("   2. Copy contents of 'deploy_supabase_manual.sql'")
    print("   3. Paste into SQL Editor and click 'Run'")
    print("   4. Wait for completion message")
    
    # Test if tables exist
    print("\n🔍 Checking current database state...")
    
    try:
        # Try to query documents table
        result = supabase.table("documents").select("id").limit(1).execute()
        print("✅ Table 'documents' exists!")
    except Exception as e:
        if "Could not find the table" in str(e):
            print("❌ Table 'documents' does not exist - deployment needed")
        else:
            print(f"⚠️  Unexpected error: {e}")
    
    try:
        # Try to query chunks table
        result = supabase.table("chunks").select("id").limit(1).execute()
        print("✅ Table 'chunks' exists!")
    except Exception as e:
        if "Could not find the table" in str(e):
            print("❌ Table 'chunks' does not exist - deployment needed")
        else:
            print(f"⚠️  Unexpected error: {e}")
    
    try:
        # Try to query sessions table
        result = supabase.table("sessions").select("id").limit(1).execute()
        print("✅ Table 'sessions' exists!")
    except Exception as e:
        if "Could not find the table" in str(e):
            print("❌ Table 'sessions' does not exist - deployment needed")
        else:
            print(f"⚠️  Unexpected error: {e}")
    
    try:
        # Try to query messages table
        result = supabase.table("messages").select("id").limit(1).execute()
        print("✅ Table 'messages' exists!")
    except Exception as e:
        if "Could not find the table" in str(e):
            print("❌ Table 'messages' does not exist - deployment needed")
        else:
            print(f"⚠️  Unexpected error: {e}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    
print("\n" + "=" * 50)
print("📚 Please use the Supabase Dashboard SQL Editor to deploy the schema")
print("   This ensures all extensions, functions, and indexes are properly created")