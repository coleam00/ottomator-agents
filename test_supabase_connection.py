#!/usr/bin/env python3
"""Test Supabase database connection and show configuration."""
import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client

def test_connection():
    """Test Supabase connection using environment variables."""
    load_dotenv()
    
    # Get configuration
    db_provider = os.getenv("DB_PROVIDER", "postgres")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    print(f"DB Provider: {db_provider}")
    print(f"Supabase URL: {supabase_url}")
    print(f"Service Key: {'Set' if supabase_service_key else 'Not set'}")
    
    if db_provider != "supabase":
        print("\n⚠️  DB_PROVIDER is not set to 'supabase'")
        return False
    
    if not supabase_url or not supabase_service_key:
        print("\n❌ Missing Supabase configuration")
        return False
    
    try:
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_service_key)
        
        # Test connection with a simple query
        response = supabase.table("chunks").select("id").limit(1).execute()
        print(f"\n✅ Connection successful!")
        print(f"   Chunks table exists: {'✓' if response else '✓'}")
        
        # Count existing chunks
        count_response = supabase.table("chunks").select("*", count="exact").execute()
        chunk_count = count_response.count if hasattr(count_response, 'count') else 0
        print(f"   Existing chunks: {chunk_count}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)