"""
Example test file for database connections.
Copy this to tests/ directory and update with your credentials.

IMPORTANT: Never commit actual credentials to version control!
Use environment variables or a .env file for sensitive data.
"""

import os
import asyncio
import asyncpg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def test_supabase_connection():
    """Test Supabase PostgreSQL connection using environment variables."""
    
    # Get configuration from environment
    supabase_url = os.getenv("SUPABASE_URL", "")
    project_ref = supabase_url.replace("https://", "").split(".")[0] if supabase_url else ""
    
    # IMPORTANT: Always get password from environment variable
    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    
    if not db_password:
        print("❌ SUPABASE_DB_PASSWORD not set in environment variables")
        print("   Please set it in your .env file or environment")
        return None
    
    # Common Supabase connection patterns
    connection_strings = [
        # Direct connection (port 5432)
        f"postgresql://postgres.{project_ref}:{db_password}@aws-0-us-west-1.pooler.supabase.com:5432/postgres",
        # Pooler connection (port 6543) 
        f"postgresql://postgres.{project_ref}:{db_password}@aws-0-us-west-1.pooler.supabase.com:6543/postgres",
        # Direct to database
        f"postgresql://postgres:{db_password}@db.{project_ref}.supabase.co:5432/postgres",
    ]
    
    for i, conn_str in enumerate(connection_strings, 1):
        print(f"\nTrying connection {i}...")
        # Mask password in output for security
        safe_conn_str = conn_str.replace(db_password, "***")
        print(f"   Connection: {safe_conn_str[:80]}...")
        
        try:
            # Test connection
            conn = await asyncpg.connect(conn_str, timeout=5)
            
            # Test query
            result = await conn.fetchval("SELECT version()")
            print(f"✅ SUCCESS with connection {i}!")
            print(f"   Database version: {result[:50]}...")
            
            await conn.close()
            return conn_str
            
        except Exception as e:
            print(f"❌ Failed: {str(e)[:100]}")
    
    return None

if __name__ == "__main__":
    result = asyncio.run(test_supabase_connection())
    if result:
        # Don't print the actual connection string with password
        print("\n✅ Connection successful!")
        print("   Check your .env file for the connection details")
    else:
        print("\n❌ All connection attempts failed")
        print("   Please check your credentials and network connection")