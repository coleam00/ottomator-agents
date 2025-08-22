#!/usr/bin/env python3
"""
Supabase Database Schema Verification Script
============================================

This script verifies that the Medical RAG database schema has been properly
deployed to Supabase. It uses the same Supabase client configuration as the
main application.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()


class SupabaseVerifier:
    """Verifies Supabase database schema deployment."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable not set")
        if not self.supabase_key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable not set")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    def test_connection(self) -> bool:
        """Test basic Supabase connection."""
        try:
            # Test connection with a simple RPC call or health check
            # Since we don't have direct access to system tables, we'll try to access a regular table
            # This will show us the error message which tells us if tables exist
            response = self.client.table("documents").select("id").limit(1).execute()
            print("✅ Supabase connection and documents table access successful")
            return True
        except Exception as e:
            error_msg = str(e)
            if "Could not find the table" in error_msg:
                print("✅ Supabase connection successful (but schema not deployed)")
                print(f"   Error: {error_msg}")
                return True  # Connection works, just no schema
            else:
                print(f"❌ Supabase connection failed: {e}")
                return False
    
    def check_extensions(self) -> Dict[str, bool]:
        """Check if required PostgreSQL extensions are installed."""
        print("\n🔧 Checking PostgreSQL Extensions...")
        results = {}
        
        required_extensions = ['vector', 'uuid-ossp', 'pg_trgm']
        
        try:
            for ext in required_extensions:
                # Use RPC to query pg_extension
                response = self.client.rpc("sql", {
                    "query": f"SELECT extname FROM pg_extension WHERE extname = '{ext}'"
                }).execute()
                
                if response.data:
                    print(f"✅ Extension '{ext}' is installed")
                    results[ext] = True
                else:
                    print(f"❌ Extension '{ext}' is NOT installed")
                    results[ext] = False
                    
        except Exception as e:
            print(f"⚠️  Could not check extensions (may need SQL RPC function): {e}")
            # Try alternative method - check if vector operations work
            try:
                response = self.client.rpc("sql", {
                    "query": "SELECT array_fill(0.1, ARRAY[768])::vector;"
                }).execute()
                print("✅ Vector extension appears to be working")
                results['vector'] = True
            except:
                print("❌ Vector extension not available")
                results['vector'] = False
        
        return results
    
    def check_tables(self) -> Dict[str, bool]:
        """Check if required tables exist."""
        print("\n📋 Checking Database Tables...")
        results = {}
        
        required_tables = ['documents', 'chunks', 'sessions', 'messages']
        
        for table in required_tables:
            try:
                response = self.client.table(table).select("*").limit(1).execute()
                print(f"✅ Table '{table}' exists and is accessible")
                results[table] = True
            except Exception as e:
                print(f"❌ Table '{table}' is NOT accessible: {str(e)[:100]}...")
                results[table] = False
        
        return results
    
    def check_functions(self) -> Dict[str, bool]:
        """Check if required database functions exist."""
        print("\n⚙️  Checking Database Functions...")
        results = {}
        
        required_functions = [
            ('match_chunks', "SELECT * FROM match_chunks('[]'::vector, 1)"),
            ('hybrid_search', "SELECT * FROM hybrid_search('[]'::vector, 'test', 1)"),
            ('get_document_chunks', "SELECT * FROM get_document_chunks('00000000-0000-0000-0000-000000000000'::uuid)"),
            ('list_documents_with_chunk_count', "SELECT * FROM list_documents_with_chunk_count(1, 0)"),
            ('get_database_stats', "SELECT * FROM get_database_stats()")
        ]
        
        for func_name, test_query in required_functions:
            try:
                # Try calling the function
                if func_name == 'match_chunks':
                    response = self.client.rpc('match_chunks', {
                        'query_embedding': '[0.1]',
                        'match_count': 1
                    }).execute()
                elif func_name == 'hybrid_search':
                    response = self.client.rpc('hybrid_search', {
                        'query_embedding': '[0.1]',
                        'query_text': 'test',
                        'match_count': 1
                    }).execute()
                elif func_name == 'get_document_chunks':
                    response = self.client.rpc('get_document_chunks', {
                        'doc_id': '00000000-0000-0000-0000-000000000000'
                    }).execute()
                elif func_name == 'list_documents_with_chunk_count':
                    response = self.client.rpc('list_documents_with_chunk_count', {
                        'doc_limit': 1,
                        'doc_offset': 0
                    }).execute()
                elif func_name == 'get_database_stats':
                    response = self.client.rpc('get_database_stats').execute()
                
                print(f"✅ Function '{func_name}' exists and is callable")
                results[func_name] = True
                
            except Exception as e:
                error_msg = str(e)
                if "does not exist" in error_msg or "function" in error_msg.lower():
                    print(f"❌ Function '{func_name}' does NOT exist")
                else:
                    print(f"✅ Function '{func_name}' exists (returned expected error)")
                results[func_name] = "does not exist" not in error_msg
        
        return results
    
    def check_test_data(self) -> bool:
        """Check if test data was inserted."""
        print("\n🧪 Checking Test Data...")
        
        try:
            response = self.client.table("documents").select("*").eq("source", "system").execute()
            
            if response.data and len(response.data) > 0:
                doc = response.data[0]
                print(f"✅ Test document found: '{doc['title']}'")
                print(f"   Source: {doc['source']}")
                print(f"   Created: {doc['created_at']}")
                return True
            else:
                print("❌ No test documents found")
                return False
                
        except Exception as e:
            print(f"❌ Could not check test data: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics if function exists."""
        print("\n📊 Database Statistics...")
        
        try:
            response = self.client.rpc('get_database_stats').execute()
            
            if response.data and len(response.data) > 0:
                stats = response.data[0]
                print(f"   Documents: {stats.get('total_documents', 0)}")
                print(f"   Chunks: {stats.get('total_chunks', 0)}")
                print(f"   Sessions: {stats.get('total_sessions', 0)}")
                print(f"   Messages: {stats.get('total_messages', 0)}")
                print(f"   Chunks with embeddings: {stats.get('chunks_with_embeddings', 0)}")
                return stats
            else:
                print("   No statistics available")
                return {}
                
        except Exception as e:
            print(f"   Could not get statistics: {e}")
            return {}
    
    def run_verification(self) -> bool:
        """Run complete verification."""
        print("🏥 Medical RAG Database Verification for Supabase")
        print("=" * 50)
        print(f"Supabase URL: {self.supabase_url}")
        print("=" * 50)
        
        # Test connection
        if not self.test_connection():
            return False
        
        # Check extensions (optional - may not work without SQL RPC)
        ext_results = self.check_extensions()
        
        # Check tables (critical)
        table_results = self.check_tables()
        tables_ok = all(table_results.values())
        
        # Check functions (critical)
        func_results = self.check_functions()
        functions_ok = all(func_results.values())
        
        # Check test data
        test_data_ok = self.check_test_data()
        
        # Get statistics
        self.get_database_stats()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 VERIFICATION SUMMARY")
        print("=" * 50)
        
        if tables_ok:
            print("✅ All required tables exist and are accessible")
        else:
            print("❌ Some tables are missing or inaccessible:")
            for table, exists in table_results.items():
                if not exists:
                    print(f"   - {table}")
        
        if functions_ok:
            print("✅ All required functions exist and are callable")
        else:
            print("❌ Some functions are missing:")
            for func, exists in func_results.items():
                if not exists:
                    print(f"   - {func}")
        
        if test_data_ok:
            print("✅ Test data is present")
        else:
            print("❌ Test data is missing")
        
        all_ok = tables_ok and functions_ok
        
        if all_ok:
            print("\n🎉 Database schema is properly deployed!")
            print("✅ Your Medical RAG API should work correctly")
            print("\nNext steps:")
            print("1. Run document ingestion: python -m ingestion.ingest")
            print("2. Test the API: python cli.py")
        else:
            print("\n❌ Database schema is NOT properly deployed")
            print("\nSolution:")
            print("1. Open Supabase SQL Editor: https://supabase.com/dashboard/project/bpopugzfbokjzgawshov/sql")
            print("2. Copy and run the contents of: deploy_supabase_manual.sql")
            print("3. Re-run this verification script")
        
        return all_ok


def main():
    """Main entry point."""
    try:
        verifier = SupabaseVerifier()
        success = verifier.run_verification()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()