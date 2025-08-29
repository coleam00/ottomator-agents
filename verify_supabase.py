#!/usr/bin/env python
"""Quick verification of Supabase ingestion results."""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def main():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    client = create_client(url, key)
    
    # Get document count
    docs = client.table("documents").select("id, title, source").execute()
    print(f"✅ Documents in Supabase: {len(docs.data)}")
    
    if docs.data:
        print("\nDocuments:")
        for doc in docs.data:
            print(f"  - {doc['title']}")
    
    # Get chunk count
    chunks = client.table("chunks").select("id", count="exact").execute()
    print(f"\n✅ Total chunks: {chunks.count if hasattr(chunks, 'count') else len(chunks.data)}")
    
    # Check if all 11 documents are present
    if len(docs.data) == 11:
        print("\n🎉 SUCCESS: All 11 documents uploaded to Supabase!")
    else:
        print(f"\n⚠️  Only {len(docs.data)}/11 documents uploaded")

if __name__ == "__main__":
    main()