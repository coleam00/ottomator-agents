#!/usr/bin/env python
"""
Run Robust Document Ingestion with All Performance Optimizations

This script implements a robust ingestion system with:
- 10x faster Neo4j operations through connection pooling and batching
- Aggressive content truncation for Graphiti (2000 chars)
- Document-level checkpointing with resume capability
- Real-time monitoring dashboard
- Pre-flight validation checks
- Smart retry with exponential backoff
- Graceful degradation on failures

Usage:
    python run_robust_ingestion.py                    # Run with defaults
    python run_robust_ingestion.py --clean           # Clean databases first
    python run_robust_ingestion.py --resume SESSION  # Resume from checkpoint
    python run_robust_ingestion.py --no-monitoring   # Disable monitoring dashboard
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.robust_ingest import RobustIngestionPipeline
from agent.models import IngestionConfig


def print_banner():
    """Print welcome banner."""
    print("="*70)
    print("🚀 ROBUST DOCUMENT INGESTION SYSTEM")
    print("="*70)
    print("Features:")
    print("  ✅ 10x faster Neo4j operations with optimizations")
    print("  ✅ Aggressive content truncation (2000 chars)")
    print("  ✅ Document checkpointing and resume capability")
    print("  ✅ Real-time monitoring dashboard")
    print("  ✅ Pre-flight validation checks")
    print("  ✅ Smart retry with exponential backoff")
    print("  ✅ Graceful degradation on failures")
    print("="*70)
    print()


def confirm_action(message: str) -> bool:
    """Get user confirmation."""
    response = input(f"{message} (yes/no): ").lower().strip()
    return response in ['yes', 'y']


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run robust document ingestion with all optimizations"
    )
    
    # Document options
    parser.add_argument(
        "--documents", "-d",
        default="medical_docs",
        help="Path to documents folder (default: medical_docs)"
    )
    
    # Database options
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean databases before ingestion"
    )
    
    # Chunking options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size in characters (default: 800)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)"
    )
    
    # Performance options
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable real-time monitoring dashboard"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip pre-flight validation checks"
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable checkpointing (not recommended)"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip knowledge graph building (faster but less complete)"
    )
    
    # Resume options
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from session ID"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimize output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"robust_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Check documents folder
    if not os.path.exists(args.documents):
        print(f"❌ Error: Documents folder not found: {args.documents}")
        return 1
    
    # Count documents
    doc_patterns = ["*.md", "*.markdown", "*.txt"]
    documents = []
    for pattern in doc_patterns:
        documents.extend(Path(args.documents).rglob(pattern))
    
    if not documents:
        print(f"❌ Error: No documents found in {args.documents}")
        return 1
    
    print(f"📄 Found {len(documents)} documents to process")
    
    # Estimate time
    estimated_time = len(documents) * 30  # ~30 seconds per document
    print(f"⏱️  Estimated time: {estimated_time/60:.1f} minutes")
    print()
    
    # Get confirmation
    if args.clean:
        print("⚠️  WARNING: --clean will delete all existing data!")
        if not confirm_action("Are you sure you want to clean databases?"):
            print("Aborted.")
            return 0
    
    if not args.quiet:
        if not confirm_action("Do you want to proceed with ingestion?"):
            print("Aborted.")
            return 0
    
    print()
    print("Starting robust ingestion...")
    print()
    
    # Create configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunk_size=args.chunk_size * 2,
        use_semantic_chunking=False,  # Disabled for speed
        extract_entities=True,
        skip_graph_building=args.skip_graph
    )
    
    # Create pipeline
    pipeline = RobustIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        enable_monitoring=not args.no_monitoring,
        enable_validation=not args.no_validation,
        enable_checkpointing=not args.no_checkpointing,
        resume_session=args.resume
    )
    
    # Run ingestion
    try:
        start_time = datetime.now()
        
        results = await pipeline.ingest_documents()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print results
        if results:
            successful = sum(1 for r in results if not r.errors)
            failed = sum(1 for r in results if r.errors)
            
            print()
            print("="*70)
            print("✅ INGESTION COMPLETE")
            print("="*70)
            print(f"Total Time: {total_time/60:.1f} minutes")
            print(f"Documents: {len(results)} processed")
            print(f"  ✅ Successful: {successful}")
            print(f"  ❌ Failed: {failed}")
            print(f"Total Chunks: {sum(r.chunks_created for r in results)}")
            print(f"Total Episodes: {sum(r.relationships_created for r in results)}")
            print(f"Total Entities: {sum(r.entities_extracted for r in results)}")
            
            if failed > 0:
                print()
                print("Failed Documents:")
                for result in results:
                    if result.errors:
                        print(f"  ❌ {result.title}")
                        for error in result.errors[:2]:
                            print(f"      {error}")
            
            print()
            print(f"Session ID: {pipeline.session_id}")
            print(f"Checkpoint saved to: .ingestion_checkpoints/checkpoint_{pipeline.session_id}.json")
            
            if not args.no_monitoring:
                print(f"Metrics exported to: metrics_{pipeline.session_id}.json")
            
            return 0
            
        else:
            print("❌ No documents were processed")
            return 1
            
    except KeyboardInterrupt:
        print()
        print("⚠️  Ingestion interrupted by user")
        print(f"Session ID: {pipeline.session_id}")
        print(f"Resume with: python run_robust_ingestion.py --resume {pipeline.session_id}")
        return 130
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        print()
        print(f"❌ Ingestion failed: {e}")
        print(f"Check log file for details")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)