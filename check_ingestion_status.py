#!/usr/bin/env python
"""Check the current status of document ingestion in the database."""

import asyncio
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.unified_db_utils import initialize_database, close_database, execute_query


async def check_ingestion_status():
    """Check what documents are currently in the database."""
    try:
        # Initialize database
        await initialize_database()
        
        # Check documents
        print("\n=== CURRENT INGESTION STATUS ===\n")
        
        # Get document count and list
        doc_query = """
            SELECT id, title, source, 
                   COALESCE(metadata->>'ingestion_date', 'Unknown') as ingestion_date,
                   LENGTH(content) as content_length
            FROM documents 
            ORDER BY source
        """
        documents = await execute_query(doc_query)
        
        print(f"Total documents in database: {len(documents)}\n")
        
        if documents:
            print("Documents ingested:")
            print("-" * 80)
            for doc in documents:
                print(f"ID: {doc['id'][:8]}...")
                print(f"Title: {doc['title']}")
                print(f"Source: {doc['source']}")
                print(f"Ingestion Date: {doc['ingestion_date']}")
                print(f"Content Length: {doc['content_length']:,} chars")
                
                # Get chunk count for this document
                chunk_query = """
                    SELECT COUNT(*) as chunk_count 
                    FROM chunks 
                    WHERE document_id = $1
                """
                chunk_result = await execute_query(chunk_query, doc['id'])
                chunk_count = chunk_result[0]['chunk_count'] if chunk_result else 0
                print(f"Chunks: {chunk_count}")
                print("-" * 80)
        
        # Get total chunk count
        total_chunks_query = "SELECT COUNT(*) as total FROM chunks"
        total_result = await execute_query(total_chunks_query)
        total_chunks = total_result[0]['total'] if total_result else 0
        
        print(f"\nTotal chunks in database: {total_chunks}")
        
        # Check for documents in medical_docs that are NOT in database
        print("\n=== MISSING DOCUMENTS ===\n")
        
        medical_docs_path = "medical_docs"
        all_docs = []
        if os.path.exists(medical_docs_path):
            for file in os.listdir(medical_docs_path):
                if file.endswith('.md'):
                    all_docs.append(file)
        
        all_docs.sort()
        
        # Get sources from database
        ingested_sources = [doc['source'] for doc in documents] if documents else []
        
        missing_docs = []
        for doc_file in all_docs:
            if doc_file not in ingested_sources:
                missing_docs.append(doc_file)
        
        if missing_docs:
            print(f"Found {len(missing_docs)} documents NOT in database:")
            for doc in missing_docs:
                print(f"  - {doc}")
        else:
            print("All documents from medical_docs/ are in the database!")
        
        print(f"\nExpected: {len(all_docs)} documents")
        print(f"Actual: {len(documents)} documents")
        print(f"Missing: {len(missing_docs)} documents")
        
    except Exception as e:
        print(f"Error checking ingestion status: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(check_ingestion_status())