#!/usr/bin/env python3
"""
Test script for Neo4j bulk ingestion functionality.
Verifies that the bulk ingestion process works correctly.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4
import logging

from dotenv import load_dotenv
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.nodes import EpisodeType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def test_bulk_ingestion():
    """Test the bulk ingestion functionality with sample data."""
    
    logger.info("=" * 60)
    logger.info("TESTING NEO4J BULK INGESTION")
    logger.info("=" * 60)
    
    # Import graph utilities
    from agent.graph_utils import GraphitiClient
    
    # Initialize graph client
    graph_client = GraphitiClient()
    
    try:
        # Initialize the client
        await graph_client.initialize()
        logger.info("✓ Graph client initialized")
        
        # Create sample episodes for testing
        test_episodes = []
        reference_time = datetime.now(timezone.utc)
        
        # Create 5 test episodes with medical content
        test_data = [
            {
                "title": "Diabetes Management",
                "content": "Type 2 diabetes is a chronic condition that affects blood sugar levels. Management includes diet, exercise, and medication. Regular monitoring is essential."
            },
            {
                "title": "Hypertension Treatment",
                "content": "High blood pressure can be managed through lifestyle changes and medication. ACE inhibitors and beta-blockers are common treatments."
            },
            {
                "title": "COVID-19 Vaccines",
                "content": "mRNA vaccines have shown high efficacy against COVID-19. Booster doses are recommended for continued protection."
            },
            {
                "title": "Mental Health Awareness",
                "content": "Depression and anxiety are common mental health conditions. Treatment options include therapy, medication, and lifestyle modifications."
            },
            {
                "title": "Preventive Care",
                "content": "Regular health screenings can detect diseases early. Annual check-ups, vaccinations, and healthy lifestyle choices are key to prevention."
            }
        ]
        
        for i, data in enumerate(test_data):
            episode = RawEpisode(
                name=f"test_episode_{i}_{uuid4().hex[:8]}",
                content=f"Document: {data['title']}\n\n{data['content']}",
                source=EpisodeType.text,
                source_description=f"Test medical document: {data['title']}",
                reference_time=reference_time
            )
            test_episodes.append(episode)
        
        logger.info(f"Created {len(test_episodes)} test episodes")
        
        # Test bulk ingestion
        logger.info("\nTesting bulk ingestion...")
        result = await graph_client.add_episodes_bulk(
            bulk_episodes=test_episodes,
            group_id="0"  # Shared knowledge base
        )
        
        if result["success"]:
            logger.info(f"✅ Bulk ingestion successful!")
            logger.info(f"   Episodes added: {result['episodes_added']}")
        else:
            logger.error(f"❌ Bulk ingestion failed!")
            logger.error(f"   Errors: {result['errors']}")
            return False
        
        # Test searching for the ingested content
        logger.info("\nTesting search functionality...")
        
        test_queries = [
            "diabetes management",
            "hypertension treatment",
            "COVID vaccines",
            "mental health",
            "preventive care"
        ]
        
        successful_searches = 0
        for query in test_queries:
            results = await graph_client.search(query, group_ids=["0"])
            if results:
                logger.info(f"✓ Found {len(results)} results for '{query}'")
                successful_searches += 1
                # Show first result
                if results:
                    first_result = results[0]
                    logger.debug(f"  Sample result: {first_result.get('fact', '')[:100]}...")
            else:
                logger.warning(f"✗ No results for '{query}'")
        
        logger.info(f"\nSearch success rate: {successful_searches}/{len(test_queries)}")
        
        # Get graph statistics
        logger.info("\nChecking graph statistics...")
        stats = await graph_client.get_graph_statistics()
        logger.info(f"Graph stats: {stats}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        
        if result["success"] and successful_searches > 0:
            logger.info("✅ All tests passed successfully!")
            logger.info("   - Bulk ingestion: PASSED")
            logger.info("   - Search functionality: PASSED")
            logger.info("   - Graph connectivity: PASSED")
            return True
        else:
            logger.error("❌ Some tests failed")
            if not result["success"]:
                logger.error("   - Bulk ingestion: FAILED")
            if successful_searches == 0:
                logger.error("   - Search functionality: FAILED")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        await graph_client.close()
        logger.info("\nGraph client closed")


async def test_full_pipeline():
    """Test the full bulk ingestion pipeline with Supabase data."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING FULL BULK INGESTION PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Import the bulk ingestion module
        from neo4j_bulk_ingestion import Neo4jBulkIngestion
        
        ingestion = Neo4jBulkIngestion()
        
        # Test fetching documents
        logger.info("\n1. Testing document fetching from Supabase...")
        documents_data = await ingestion.fetch_all_documents_and_chunks()
        
        if documents_data:
            logger.info(f"✓ Successfully fetched {len(documents_data)} documents")
            total_chunks = sum(len(d["chunks"]) for d in documents_data)
            logger.info(f"  Total chunks: {total_chunks}")
        else:
            logger.warning("✗ No documents fetched")
            return False
        
        # Test episode preparation (use only first document for testing)
        logger.info("\n2. Testing episode preparation...")
        test_data = documents_data[:1]  # Use only first document for test
        bulk_episodes = ingestion.prepare_bulk_episodes(test_data)
        
        if bulk_episodes:
            logger.info(f"✓ Successfully prepared {len(bulk_episodes)} episodes")
        else:
            logger.error("✗ Failed to prepare episodes")
            return False
        
        # Test bulk ingestion with small batch
        logger.info("\n3. Testing bulk ingestion with small batch...")
        results = await ingestion.perform_bulk_ingestion(
            bulk_episodes[:5],  # Test with only 5 episodes
            batch_size=5
        )
        
        if results["successful_batches"] > 0:
            logger.info(f"✓ Successfully ingested {results['successful_batches']} batches")
        else:
            logger.error("✗ No batches successfully ingested")
            return False
        
        logger.info("\n✅ Full pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test execution."""
    
    print("\n" + "="*60)
    print("NEO4J BULK INGESTION TEST SUITE")
    print("="*60)
    print("\nThis will test:")
    print("1. Basic bulk ingestion with sample data")
    print("2. Search functionality after ingestion")
    print("3. Full pipeline with Supabase integration (optional)")
    print("\n")
    
    # Run basic test
    logger.info("Starting basic bulk ingestion test...")
    basic_test_passed = await test_bulk_ingestion()
    
    if basic_test_passed:
        # Ask if user wants to run full pipeline test
        print("\n" + "="*60)
        response = input("\nBasic test passed! Run full pipeline test? (yes/no): ").strip().lower()
        if response == "yes":
            pipeline_test_passed = await test_full_pipeline()
            
            if pipeline_test_passed:
                logger.info("\n🎉 All tests passed successfully!")
            else:
                logger.error("\n⚠️ Pipeline test failed")
        else:
            logger.info("\nSkipping full pipeline test")
    else:
        logger.error("\n⚠️ Basic test failed - skipping pipeline test")
    
    logger.info("\nTest suite completed")


if __name__ == "__main__":
    asyncio.run(main())