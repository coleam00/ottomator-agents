"""Test database connection"""
import os
import asyncio
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def test_supabase_connection():
    """Test Supabase PostgreSQL connection"""
    
    # Try different connection formats
    supabase_url = os.getenv("SUPABASE_URL", "").replace("https://", "")
    project_ref = supabase_url.split(".")[0] if supabase_url else "bpopugzfbokjzgawshov"
    
    # Common Supabase connection patterns
    connection_strings = [
        # Direct connection (port 5432)
        f"postgresql://postgres.{project_ref}:zwxx5r5rEN3LmqNhhQfZQEQcihekvH4AED4FrLTdG9I@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
        # Pooler connection (port 6543) 
        f"postgresql://postgres.{project_ref}:zwxx5r5rEN3LmqNhhQfZQEQcihekvH4AED4FrLTdG9I@aws-0-us-west-1.pooler.supabase.com:6543/postgres",
        # Session pooler
        f"postgresql://postgres.{project_ref}:zwxx5r5rEN3LmqNhhQfZQEQcihekvH4AED4FrLTdG9I@aws-0-us-west-1.pooler.supabase.com:5432/postgres?pgbouncer=true",
        # Direct to database
        f"postgresql://postgres:zwxx5r5rEN3LmqNhhQfZQEQcihekvH4AED4FrLTdG9I@db.{project_ref}.supabase.co:5432/postgres",
    ]
    
    for i, conn_str in enumerate(connection_strings, 1):
        print(f"\nTrying connection {i}...")
        try:
            # Test connection
            conn = await asyncpg.connect(conn_str, timeout=5)
            
            # Test query
            result = await conn.fetchval("SELECT version()")
            print(f"✅ SUCCESS with connection {i}!")
            print(f"   Database version: {result[:50]}...")
            print(f"   Connection string: {conn_str[:80]}...")
            
            await conn.close()
            return conn_str
            
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}")
    
    return None

if __name__ == "__main__":
    result = asyncio.run(test_supabase_connection())
    if result:
        print(f"\n✅ Working connection string:\n{result}")
        print("\nAdd this to your .env file as DATABASE_URL")
    else:
        print("\n❌ All connection attempts failed")