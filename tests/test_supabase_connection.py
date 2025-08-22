#!/usr/bin/env python3
"""
Test script to validate Supabase connection and configuration.
"""

import os
import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_supabase_setup():
    """Test Supabase configuration and connection."""
    print("🔧 Testing Supabase Configuration...")
    print("=" * 50)
    
    # Check environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    db_provider = os.getenv("DB_PROVIDER", "postgres")
    
    print(f"📊 Database Provider: {db_provider}")
    print(f"🌐 Supabase URL: {supabase_url}")
    print(f"🔑 Anon Key: {'✅ Set' if supabase_anon_key else '❌ Missing'}")
    print(f"🔑 Service Role Key: {'✅ Set' if supabase_service_key else '❌ Missing'}")
    print()
    
    # Check if running with Supabase provider
    if db_provider != "supabase":
        print(f"⚠️  Warning: DB_PROVIDER is set to '{db_provider}', not 'supabase'")
        print("   To test Supabase, set DB_PROVIDER=supabase in your .env file")
        print()
    
    # Test imports
    try:
        print("📦 Testing imports...")
        from agent.unified_db_utils import (
            test_connection,
            get_provider_info,
            validate_configuration,
            health_check
        )
        print("✅ Imports successful")
        print()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Make sure you've installed: pip install supabase==2.10.0")
        return False
    
    # Test provider info
    try:
        print("🔍 Getting provider information...")
        provider_info = await get_provider_info()
        for key, value in provider_info.items():
            print(f"   {key}: {value}")
        print()
    except Exception as e:
        print(f"❌ Failed to get provider info: {e}")
        return False
    
    # Test configuration validation
    try:
        print("✅ Validating configuration...")
        config_validation = await validate_configuration()
        print(f"   Provider: {config_validation['provider']}")
        print(f"   Configuration Valid: {config_validation['configuration_valid']}")
        print(f"   Connection OK: {config_validation['connection_ok']}")
        
        if config_validation['issues']:
            print("   Issues:")
            for issue in config_validation['issues']:
                print(f"   ❌ {issue}")
        print()
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test database connection
    try:
        print("🔌 Testing database connection...")
        connection_ok = await test_connection()
        if connection_ok:
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
            return False
        print()
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False
    
    # Test health check
    try:
        print("🏥 Running health check...")
        health = await health_check()
        print(f"   Status: {health['status']}")
        print(f"   Provider: {health['provider']}")
        print(f"   Connection: {health.get('connection', 'unknown')}")
        
        if 'stats' in health and health['stats']:
            print("   Database Stats:")
            for key, value in health['stats'].items():
                print(f"     {key}: {value}")
        
        if health.get('error'):
            print(f"   ❌ Error: {health['error']}")
            return False
        print()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test specific Supabase operations
    if db_provider == "supabase":
        try:
            print("🧪 Testing Supabase-specific operations...")
            
            # Test database stats
            from agent.unified_db_utils import get_database_stats
            stats = await get_database_stats()
            print("   Database Statistics:")
            for key, value in stats.items():
                print(f"     {key}: {value}")
            print()
            
        except Exception as e:
            print(f"❌ Supabase operations test failed: {e}")
            return False
    
    print("🎉 All tests passed! Supabase is properly configured.")
    
    # Next steps
    print("\n📋 Next Steps:")
    print("1. Run the API server: python -m agent.api")
    print("2. Test the health endpoint: curl http://localhost:8058/health")
    print("3. Check provider info: curl http://localhost:8058/provider/info")
    print("4. Run document ingestion if you haven't already")
    print("5. Test the agent with: python cli.py")
    
    return True


async def test_vector_operations():
    """Test vector operations with a sample embedding."""
    print("\n🔢 Testing Vector Operations...")
    print("=" * 50)
    
    try:
        from agent.unified_db_utils import vector_search, get_database_stats
        
        # Check if we have any chunks with embeddings
        stats = await get_database_stats()
        chunk_count = stats.get('chunks', 0)
        
        if chunk_count == 0:
            print("⚠️  No chunks found in database. Run ingestion first:")
            print("   python -m ingestion.ingest")
            return False
        
        print(f"📊 Found {chunk_count} chunks in database")
        
        # Create a sample embedding (3072 dimensions for Gemini)
        print("🧮 Creating sample embedding for test...")
        sample_embedding = [0.1] * 3072  # Simple test embedding
        
        # Test vector search
        print("🔍 Testing vector search...")
        results = await vector_search(sample_embedding, limit=3)
        
        if results:
            print(f"✅ Vector search successful! Found {len(results)} results")
            for i, result in enumerate(results[:2]):  # Show first 2 results
                print(f"   Result {i+1}:")
                print(f"     Document: {result.get('document_title', 'Unknown')}")
                print(f"     Similarity: {result.get('similarity', 0):.4f}")
                print(f"     Content preview: {result.get('content', '')[:100]}...")
                print()
        else:
            print("⚠️  Vector search returned no results")
            print("   This might be normal if embeddings haven't been generated yet")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        print("   Make sure you've run document ingestion first")
        return False


async def main():
    """Main test function."""
    print("🚀 Supabase Configuration Test")
    print("=" * 50)
    print()
    
    # Test basic setup
    basic_test_passed = await test_supabase_setup()
    
    if not basic_test_passed:
        print("\n❌ Basic configuration test failed!")
        print("   Please check your Supabase configuration and try again.")
        sys.exit(1)
    
    # Test vector operations if basic test passed
    vector_test_passed = await test_vector_operations()
    
    if basic_test_passed and vector_test_passed:
        print("\n🎉 All tests completed successfully!")
        print("   Your Supabase configuration is ready for production use.")
    elif basic_test_passed:
        print("\n✅ Basic configuration test passed!")
        print("   Vector operations test skipped (run ingestion first)")
    
    print("\n📖 For more information, see: SUPABASE_MIGRATION_GUIDE.md")


if __name__ == "__main__":
    asyncio.run(main())