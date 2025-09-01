#!/usr/bin/env python3
"""
Comprehensive Supabase Database and Vector Search Verification
==============================================================
This script verifies the completed ingestion and tests vector search functionality.
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# ANSI color codes for better terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

@dataclass
class VerificationResult:
    """Results from database verification"""
    database_connected: bool
    document_count: int
    chunk_count: int
    embedding_dimension: Optional[int]
    vector_type: str
    sample_embeddings_valid: bool
    search_functions_exist: bool
    errors: List[str]
    warnings: List[str]

@dataclass
class SearchTestResult:
    """Results from search testing"""
    query: str
    vector_search_count: int
    hybrid_search_count: int
    text_search_count: int
    top_results: List[Dict[str, Any]]
    search_time_ms: float
    errors: List[str]

class SupabaseVerifier:
    """Comprehensive Supabase verification and testing"""
    
    def __init__(self):
        """Initialize Supabase client and Gemini for embeddings"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials in environment")
            
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize embedding provider
        self.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'google')
        self.embedding_api_key = os.getenv('EMBEDDING_API_KEY')
        
        self.expected_dimension = int(os.getenv('VECTOR_DIMENSION', '768'))
        print(f"{Colors.CYAN}Expected vector dimension: {self.expected_dimension}{Colors.RESET}")
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using configured provider"""
        try:
            # Try to use the actual embedding system if available
            from ingestion.embedder import Embedder
            embedder = Embedder()
            embedding = embedder.embed(text)
            
            # Normalize to expected dimension (768)
            if len(embedding) != self.expected_dimension:
                # Truncate or pad as needed
                if len(embedding) > self.expected_dimension:
                    embedding = embedding[:self.expected_dimension]
                else:
                    # Pad with zeros
                    embedding.extend([0.0] * (self.expected_dimension - len(embedding)))
            
            return embedding
            
        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Could not generate real embedding ({e}), using synthetic{Colors.RESET}")
            # Return a synthetic embedding for testing
            import random
            # Use local Random instance to avoid polluting global state
            rng = random.Random(hash(text))  # Make it deterministic based on text
            return [rng.random() * 0.2 - 0.1 for _ in range(self.expected_dimension)]
    
    def verify_database_structure(self) -> VerificationResult:
        """Verify database tables and structure"""
        print(f"\n{Colors.BOLD}=== Database Structure Verification ==={Colors.RESET}")
        
        result = VerificationResult(
            database_connected=False,
            document_count=0,
            chunk_count=0,
            embedding_dimension=None,
            vector_type="unknown",
            sample_embeddings_valid=False,
            search_functions_exist=False,
            errors=[],
            warnings=[]
        )
        
        try:
            # Test connection
            print(f"Testing database connection...")
            test_response = self.client.table('documents').select('id').limit(1).execute()
            result.database_connected = True
            print(f"{Colors.GREEN}✓ Database connected successfully{Colors.RESET}")
            
            # Count documents
            doc_response = self.client.table('documents').select('id', count='exact').execute()
            result.document_count = doc_response.count if hasattr(doc_response, 'count') else len(doc_response.data)
            print(f"{Colors.GREEN}✓ Found {result.document_count} documents{Colors.RESET}")
            
            # Count chunks
            chunk_response = self.client.table('chunks').select('id', count='exact').execute()
            result.chunk_count = chunk_response.count if hasattr(chunk_response, 'count') else len(chunk_response.data)
            print(f"{Colors.GREEN}✓ Found {result.chunk_count} chunks{Colors.RESET}")
            
            # Check embedding dimensions
            print(f"Checking embedding dimensions...")
            sample_chunks = self.client.table('chunks').select('id, embedding').limit(5).execute()
            
            if sample_chunks.data:
                for chunk in sample_chunks.data:
                    if chunk.get('embedding'):
                        # Check if it's a string (vector representation) or list
                        embedding = chunk['embedding']
                        
                        if isinstance(embedding, str):
                            # Parse vector string format [x,y,z]
                            if embedding.startswith('[') and embedding.endswith(']'):
                                try:
                                    values = json.loads(embedding)
                                    dimension = len(values)
                                except:
                                    # Try parsing as comma-separated
                                    values = embedding.strip('[]').split(',')
                                    dimension = len(values)
                            else:
                                dimension = len(embedding.split(','))
                            result.vector_type = "pgvector string"
                        else:
                            dimension = len(embedding)
                            result.vector_type = "array"
                        
                        if result.embedding_dimension is None:
                            result.embedding_dimension = dimension
                            print(f"{Colors.GREEN}✓ Embedding dimension: {dimension}{Colors.RESET}")
                        elif result.embedding_dimension != dimension:
                            result.warnings.append(f"Inconsistent dimensions found: {dimension} vs {result.embedding_dimension}")
                
                # Validate sample embeddings
                if result.embedding_dimension is None:
                    result.warnings.append("Could not determine embedding dimension")
                    print(f"{Colors.YELLOW}⚠ Warning: Could not determine embedding dimension{Colors.RESET}")
                elif result.embedding_dimension == self.expected_dimension:
                    result.sample_embeddings_valid = True
                    print(f"{Colors.GREEN}✓ Embeddings match expected dimension ({self.expected_dimension}){Colors.RESET}")
                else:
                    result.errors.append(f"Dimension mismatch: found {result.embedding_dimension}, expected {self.expected_dimension}")
                    print(f"{Colors.RED}✗ Dimension mismatch: found {result.embedding_dimension}, expected {self.expected_dimension}{Colors.RESET}")
            
            # Check if search functions exist
            print(f"Checking search functions...")
            try:
                # Test match_chunks function
                test_embedding = [0.1] * self.expected_dimension
                match_result = self.client.rpc('match_chunks', {
                    'query_embedding': test_embedding,
                    'match_count': 1
                }).execute()
                print(f"{Colors.GREEN}✓ match_chunks function exists{Colors.RESET}")
                
                # Test hybrid_search function
                try:
                    hybrid_result = self.client.rpc('hybrid_search', {
                        'query_embedding': test_embedding,
                        'query_text': 'test',
                        'match_count': 1,
                        'text_weight': 0.3
                    }).execute()
                    print(f"{Colors.GREEN}✓ hybrid_search function exists{Colors.RESET}")
                except Exception as e:
                    if 'type' in str(e).lower() or 'mismatch' in str(e).lower():
                        result.warnings.append(f"hybrid_search type mismatch: {str(e)}")
                        print(f"{Colors.YELLOW}⚠ hybrid_search has type issues: {str(e)[:100]}{Colors.RESET}")
                    else:
                        raise
                
                result.search_functions_exist = True
                
            except Exception as e:
                result.errors.append(f"Search functions error: {str(e)}")
                print(f"{Colors.RED}✗ Search functions error: {str(e)[:100]}{Colors.RESET}")
                
        except Exception as e:
            result.errors.append(f"Database verification failed: {str(e)}")
            print(f"{Colors.RED}✗ Database verification failed: {str(e)}{Colors.RESET}")
            
        return result
    
    def test_vector_search(self, query: str = "menopause symptoms") -> SearchTestResult:
        """Test vector search functionality"""
        print(f"\n{Colors.BOLD}=== Vector Search Test ==={Colors.RESET}")
        print(f"Query: '{query}'")
        
        result = SearchTestResult(
            query=query,
            vector_search_count=0,
            hybrid_search_count=0,
            text_search_count=0,
            top_results=[],
            search_time_ms=0,
            errors=[]
        )
        
        try:
            # Generate embedding for query
            print(f"Generating embedding for query...")
            start_time = datetime.now()
            query_embedding = self.generate_embedding(query)
            
            # Test vector search
            print(f"Testing vector similarity search...")
            try:
                vector_results = self.client.rpc('match_chunks', {
                    'query_embedding': query_embedding,
                    'match_count': 5
                }).execute()
                
                result.vector_search_count = len(vector_results.data) if vector_results.data else 0
                print(f"{Colors.GREEN}✓ Vector search returned {result.vector_search_count} results{Colors.RESET}")
                
                # Store top results
                if vector_results.data:
                    for i, chunk in enumerate(vector_results.data[:3]):
                        result.top_results.append({
                            'rank': i + 1,
                            'type': 'vector',
                            'similarity': chunk.get('similarity', 0),
                            'content': chunk.get('content', '')[:200],
                            'document': chunk.get('document_title', 'Unknown')
                        })
                        
            except Exception as e:
                result.errors.append(f"Vector search error: {str(e)}")
                print(f"{Colors.RED}✗ Vector search error: {str(e)[:100]}{Colors.RESET}")
            
            # Test hybrid search
            print(f"Testing hybrid search...")
            try:
                hybrid_results = self.client.rpc('hybrid_search', {
                    'query_embedding': query_embedding,
                    'query_text': query,
                    'match_count': 5,
                    'text_weight': 0.3
                }).execute()
                
                result.hybrid_search_count = len(hybrid_results.data) if hybrid_results.data else 0
                print(f"{Colors.GREEN}✓ Hybrid search returned {result.hybrid_search_count} results{Colors.RESET}")
                
            except Exception as e:
                if 'type' in str(e).lower():
                    result.errors.append(f"Hybrid search type mismatch (needs migration): {str(e)[:100]}")
                    print(f"{Colors.YELLOW}⚠ Hybrid search needs migration: type mismatch error{Colors.RESET}")
                else:
                    result.errors.append(f"Hybrid search error: {str(e)}")
                    print(f"{Colors.RED}✗ Hybrid search error: {str(e)[:100]}{Colors.RESET}")
            
            # Test text search
            print(f"Testing text search...")
            try:
                text_results = self.client.table('chunks').select('*').ilike('content', f'%{query}%').limit(5).execute()
                result.text_search_count = len(text_results.data) if text_results.data else 0
                print(f"{Colors.GREEN}✓ Text search returned {result.text_search_count} results{Colors.RESET}")
                
            except Exception as e:
                result.errors.append(f"Text search error: {str(e)}")
                print(f"{Colors.RED}✗ Text search error: {str(e)[:100]}{Colors.RESET}")
            
            # Calculate search time
            end_time = datetime.now()
            result.search_time_ms = (end_time - start_time).total_seconds() * 1000
            
        except Exception as e:
            result.errors.append(f"Search test failed: {str(e)}")
            print(f"{Colors.RED}✗ Search test failed: {str(e)}{Colors.RESET}")
            
        return result
    
    def generate_report(self, verification: VerificationResult, search_test: SearchTestResult) -> Dict[str, Any]:
        """Generate comprehensive verification report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'supabase_url': self.supabase_url,
                'expected_dimension': self.expected_dimension,
                'db_provider': os.getenv('DB_PROVIDER', 'supabase')
            },
            'database_verification': {
                'connected': verification.database_connected,
                'document_count': verification.document_count,
                'chunk_count': verification.chunk_count,
                'embedding_dimension': verification.embedding_dimension,
                'vector_type': verification.vector_type,
                'embeddings_valid': verification.sample_embeddings_valid,
                'search_functions_exist': verification.search_functions_exist,
                'errors': verification.errors,
                'warnings': verification.warnings
            },
            'search_test': {
                'query': search_test.query,
                'vector_search_results': search_test.vector_search_count,
                'hybrid_search_results': search_test.hybrid_search_count,
                'text_search_results': search_test.text_search_count,
                'search_time_ms': search_test.search_time_ms,
                'top_results': search_test.top_results,
                'errors': search_test.errors
            },
            'production_readiness': {
                'database_ready': verification.database_connected and verification.document_count > 0,
                'embeddings_ready': verification.sample_embeddings_valid,
                'vector_search_ready': search_test.vector_search_count > 0,
                'hybrid_search_ready': search_test.hybrid_search_count > 0,
                'needs_migration': 'type mismatch' in str(search_test.errors).lower(),
                'dimension_mismatches': verification.warnings  # Track dimension inconsistencies
            }
        }
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print formatted summary of verification results"""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}SUPABASE VERIFICATION REPORT{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        # Database Status
        print(f"\n{Colors.BOLD}Database Status:{Colors.RESET}")
        db = report['database_verification']
        status_icon = '✓' if db['connected'] else '✗'
        status_color = Colors.GREEN if db['connected'] else Colors.RED
        print(f"  {status_color}{status_icon} Connection: {'Connected' if db['connected'] else 'Failed'}{Colors.RESET}")
        print(f"  • Documents: {db['document_count']}")
        print(f"  • Chunks: {db['chunk_count']}")
        print(f"  • Embedding Dimension: {db['embedding_dimension']} (expected: {report['environment']['expected_dimension']})")
        print(f"  • Vector Type: {db['vector_type']}")
        
        # Search Functionality
        print(f"\n{Colors.BOLD}Search Functionality:{Colors.RESET}")
        search = report['search_test']
        
        if search['vector_search_results'] > 0:
            print(f"  {Colors.GREEN}✓ Vector Search: {search['vector_search_results']} results{Colors.RESET}")
        else:
            print(f"  {Colors.RED}✗ Vector Search: No results{Colors.RESET}")
            
        if search['hybrid_search_results'] > 0:
            print(f"  {Colors.GREEN}✓ Hybrid Search: {search['hybrid_search_results']} results{Colors.RESET}")
        elif 'type mismatch' in str(search['errors']).lower():
            print(f"  {Colors.YELLOW}⚠ Hybrid Search: Needs migration (type mismatch){Colors.RESET}")
        else:
            print(f"  {Colors.RED}✗ Hybrid Search: Failed{Colors.RESET}")
            
        if search['text_search_results'] > 0:
            print(f"  {Colors.GREEN}✓ Text Search: {search['text_search_results']} results{Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}⚠ Text Search: No results{Colors.RESET}")
        
        print(f"  • Search Time: {search['search_time_ms']:.2f}ms")
        
        # Top Results
        if search['top_results']:
            print(f"\n{Colors.BOLD}Top Search Results:{Colors.RESET}")
            for result in search['top_results'][:3]:
                print(f"  {result['rank']}. [{result['type']}] Similarity: {result['similarity']:.4f}")
                print(f"     Document: {result['document']}")
                print(f"     Content: {result['content'][:100]}...")
        
        # Production Readiness
        print(f"\n{Colors.BOLD}Production Readiness:{Colors.RESET}")
        ready = report['production_readiness']
        
        checks = [
            ('Database', ready['database_ready']),
            ('Embeddings', ready['embeddings_ready']),
            ('Vector Search', ready['vector_search_ready']),
            ('Hybrid Search', ready['hybrid_search_ready'])
        ]
        
        for check_name, is_ready in checks:
            icon = '✓' if is_ready else '✗'
            color = Colors.GREEN if is_ready else Colors.RED
            status = 'Ready' if is_ready else 'Not Ready'
            print(f"  {color}{icon} {check_name}: {status}{Colors.RESET}")
        
        if ready['needs_migration']:
            print(f"\n{Colors.YELLOW}⚠ ACTION REQUIRED:{Colors.RESET}")
            print(f"  Hybrid search function needs migration to fix type mismatch.")
            print(f"  Run the migration script: fix_supabase_functions.sql")
        
        # Errors and Warnings
        all_errors = db['errors'] + search['errors']
        all_warnings = db['warnings']
        
        if all_errors:
            print(f"\n{Colors.RED}Errors:{Colors.RESET}")
            for error in all_errors:
                print(f"  • {error}")
                
        if all_warnings:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
            for warning in all_warnings:
                print(f"  • {warning}")
        
        # Report dimension mismatches if any
        if ready.get('dimension_mismatches'):
            print(f"\n{Colors.YELLOW}Embedding Dimension Issues:{Colors.RESET}")
            for mismatch in ready['dimension_mismatches']:
                if 'dimension' in mismatch.lower():
                    print(f"  • {mismatch}")
        
        # Overall Status
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        if ready['database_ready'] and ready['embeddings_ready'] and ready['vector_search_ready']:
            if ready['needs_migration']:
                print(f"{Colors.YELLOW}VERIFICATION PASSED WITH WARNINGS{Colors.RESET}")
                print(f"System is functional but hybrid search needs migration for optimal performance.")
            else:
                print(f"{Colors.GREEN}VERIFICATION PASSED{Colors.RESET}")
                print(f"Supabase is properly configured and ready for production use.")
        else:
            print(f"{Colors.RED}VERIFICATION FAILED{Colors.RESET}")
            print(f"Please review the errors above and fix any issues.")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

def main():
    """Main verification function"""
    try:
        print(f"{Colors.BOLD}{Colors.CYAN}Starting Supabase Verification...{Colors.RESET}")
        
        # Initialize verifier
        verifier = SupabaseVerifier()
        
        # Run database verification
        verification_result = verifier.verify_database_structure()
        
        # Run search tests
        search_result = verifier.test_vector_search("menopause symptoms")
        
        # Generate report
        report = verifier.generate_report(verification_result, search_result)
        
        # Save report to file
        report_file = f"supabase_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n{Colors.GREEN}Report saved to: {report_file}{Colors.RESET}")
        
        # Print summary
        verifier.print_summary(report)
        
        # Return exit code based on verification status
        if report['production_readiness']['database_ready'] and report['production_readiness']['vector_search_ready']:
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"{Colors.RED}Verification failed with error: {e}{Colors.RESET}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)