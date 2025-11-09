#!/usr/bin/env python3
"""
List all knowledge items in Archon knowledge base
Usage: python3 list_knowledge.py [archon_host]
"""
import sys
from archon_client import ArchonClient


def main():
    archon_host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8181"
    client = ArchonClient(base_url=archon_host)

    # Verify connection
    projects = client.list_projects()
    if not projects.get('success', True):
        print(f"✗ Cannot connect to Archon at {archon_host}")
        print(f"Error: {projects.get('error')}")
        return 1

    print(f"✓ Connected to Archon at {archon_host}")
    print()

    # List knowledge items
    result = client.list_knowledge_items(limit=100)

    if not result.get('success', True):
        print(f"Error listing knowledge items: {result.get('error')}")
        return 1

    items = result.get('items', [])
    total = result.get('total', 0)

    print(f"📚 Knowledge Base Summary")
    print(f"=" * 60)
    print(f"Total knowledge items: {total}")
    print()

    if items:
        # Calculate totals
        total_words = 0
        total_code_examples = 0

        for item in items:
            meta = item.get('metadata', {})
            total_words += meta.get('word_count', 0)
            total_code_examples += meta.get('code_examples_count', 0)

        # Group by source type
        by_type = {}
        for item in items:
            source_type = item.get('source_type', 'unknown')
            if source_type not in by_type:
                by_type[source_type] = []
            by_type[source_type].append(item)

        print("Knowledge items by type:")
        for source_type, type_items in sorted(by_type.items()):
            print(f"  {source_type}: {len(type_items)} items")
        print()

        print("Available knowledge items:")
        print("-" * 60)
        for idx, item in enumerate(items[:50], 1):  # Show first 50
            title = item.get('title', 'Untitled')
            source_type = item.get('source_type', 'unknown')
            url = item.get('url', '')

            # Extract metadata
            meta = item.get('metadata', {})
            words = meta.get('word_count', 0)
            pages = meta.get('estimated_pages', 0)
            code_examples = meta.get('code_examples_count', 0)
            last_scraped = meta.get('last_scraped', 'N/A')
            if last_scraped != 'N/A':
                last_scraped = last_scraped[:10]  # Date only

            print(f"{idx}. {title}")
            print(f"   Type: {source_type}")
            if url:
                print(f"   URL: {url}")
            print(f"   Content: {words:,} words (~{pages:.1f} pages)")
            print(f"   Code Examples: {code_examples:,}")
            print(f"   Last Updated: {last_scraped}")
            print()

        if total > 50:
            print(f"... and {total - 50} more items")

        print()
        print("=" * 60)
        print("TOTALS")
        print("=" * 60)
        print(f"Total Sources: {total}")
        print(f"Total Content: {total_words:,} words")
        print(f"Total Code Examples: {total_code_examples:,}")
        print(f"Equivalent Pages: ~{total_words/250:.1f} pages")
    else:
        print("No knowledge items found. The knowledge base is empty.")
        print()
        print("You can add knowledge by:")
        print("  - Crawling documentation websites")
        print("  - Uploading documents (PDF, Word, Markdown)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
