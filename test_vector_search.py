#!/usr/bin/env python3
"""
Test Vector Search Functionality for Menopause Content
======================================================
This script tests that vector search returns relevant menopause-related content.
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

class VectorSearchTester:
    """Test vector search functionality"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Missing Supabase credentials")
            
        self.client = create_client(self.supabase_url, self.supabase_key)
        self.expected_dimension = int(os.getenv('VECTOR_DIMENSION', '768'))
        
    def get_sample_embedding(self, seed_text: str = "menopause") -> List[float]:
        """Get a sample embedding from the database for testing"""
        try:
            # Try to find a chunk that contains the seed text
            result = self.client.table('chunks').select('embedding').ilike('content', f'%{seed_text}%').limit(1).execute()
            
            if result.data and result.data[0].get('embedding'):
                embedding_str = result.data[0]['embedding']
                
                # Parse the pgvector string format
                if isinstance(embedding_str, str):
                    if embedding_str.startswith('[') and embedding_str.endswith(']'):
                        try:
                            embedding = json.loads(embedding_str)
                        except:
                            # Try parsing as comma-separated
                            embedding = [float(x) for x in embedding_str.strip('[]').split(',')]
                    else:
                        embedding = [float(x) for x in embedding_str.split(',')]
                    
                    print(f"{Colors.GREEN}✓ Using real embedding from chunk containing '{seed_text}'{Colors.RESET}")
                    return embedding
                    
        except Exception as e:
            print(f"{Colors.YELLOW}Could not get sample embedding: {e}{Colors.RESET}")
        
        # Generate synthetic embedding as fallback
        import random
        random.seed(hash(seed_text))
        embedding = [random.random() * 0.2 - 0.1 for _ in range(self.expected_dimension)]
        print(f"{Colors.YELLOW}Using synthetic embedding for testing{Colors.RESET}")
        return embedding
    
    def test_search_queries(self):
        """Test various menopause-related search queries"""
        test_queries = [
            ("menopause", "General menopause information"),
            ("perimenopause", "Perimenopause transition period"),
            ("hot flashes", "Common menopause symptom"),
            ("hormone therapy", "Treatment option"),
            ("bone loss", "Health concern during menopause"),
            ("mood changes", "Emotional symptoms"),
            ("estrogen", "Key hormone"),
            ("symptoms", "General symptom search")
        ]
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}VECTOR SEARCH TEST RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        all_results = []
        
        for query_text, description in test_queries:
            print(f"\n{Colors.BOLD}Query: '{query_text}' - {description}{Colors.RESET}")
            print("-" * 50)
            
            # Get embedding for this query
            embedding = self.get_sample_embedding(query_text)
            
            try:
                # Perform vector search
                results = self.client.rpc('match_chunks', {
                    'query_embedding': embedding,
                    'match_count': 3
                }).execute()
                
                if results.data:
                    print(f"{Colors.GREEN}Found {len(results.data)} results:{Colors.RESET}")
                    
                    for i, chunk in enumerate(results.data, 1):
                        similarity = chunk.get('similarity', 0)
                        content = chunk.get('content', '')[:150]
                        doc_title = chunk.get('document_title', 'Unknown')
                        
                        # Color code based on similarity
                        if similarity > 0.8:
                            sim_color = Colors.GREEN
                        elif similarity > 0.6:
                            sim_color = Colors.YELLOW
                        else:
                            sim_color = Colors.CYAN
                        
                        print(f"\n  {i}. {Colors.BOLD}Document:{Colors.RESET} {doc_title}")
                        print(f"     {Colors.BOLD}Similarity:{Colors.RESET} {sim_color}{similarity:.4f}{Colors.RESET}")
                        print(f"     {Colors.BOLD}Content:{Colors.RESET} {content}...")
                        
                        all_results.append({
                            'query': query_text,
                            'rank': i,
                            'similarity': similarity,
                            'document': doc_title,
                            'content_preview': content
                        })
                else:
                    print(f"{Colors.RED}No results found{Colors.RESET}")
                    
            except Exception as e:
                print(f"{Colors.RED}Error during search: {e}{Colors.RESET}")
        
        # Summary statistics
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY STATISTICS{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        if all_results:
            # Find most relevant documents
            doc_counts = {}
            for result in all_results:
                doc = result['document']
                doc_counts[doc] = doc_counts.get(doc, 0) + 1
            
            print(f"\n{Colors.BOLD}Most Relevant Documents:{Colors.RESET}")
            for doc, count in sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {doc}: appeared {count} times")
            
            # Average similarity scores
            avg_similarity = sum(r['similarity'] for r in all_results) / len(all_results)
            max_similarity = max(r['similarity'] for r in all_results)
            min_similarity = min(r['similarity'] for r in all_results)
            
            print(f"\n{Colors.BOLD}Similarity Scores:{Colors.RESET}")
            print(f"  • Average: {avg_similarity:.4f}")
            print(f"  • Maximum: {max_similarity:.4f}")
            print(f"  • Minimum: {min_similarity:.4f}")
            
            # Check for menopause-related content
            menopause_keywords = ['menopause', 'perimenopause', 'hormone', 'estrogen', 'symptom', 
                                  'hot flash', 'bone', 'mood', 'therapy', 'treatment']
            
            relevant_count = 0
            for result in all_results:
                content_lower = result['content_preview'].lower()
                if any(keyword in content_lower for keyword in menopause_keywords):
                    relevant_count += 1
            
            relevance_pct = (relevant_count / len(all_results)) * 100
            
            print(f"\n{Colors.BOLD}Content Relevance:{Colors.RESET}")
            print(f"  • Results with menopause keywords: {relevant_count}/{len(all_results)} ({relevance_pct:.1f}%)")
            
            if relevance_pct >= 80:
                print(f"  {Colors.GREEN}✓ Excellent relevance - vector search is working well{Colors.RESET}")
            elif relevance_pct >= 60:
                print(f"  {Colors.YELLOW}⚠ Good relevance - vector search is functional{Colors.RESET}")
            else:
                print(f"  {Colors.RED}✗ Low relevance - vector search may need tuning{Colors.RESET}")
                
        # Save detailed results
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_queries': [q[0] for q in test_queries],
            'total_results': len(all_results),
            'detailed_results': all_results,
            'statistics': {
                'avg_similarity': avg_similarity if all_results else 0,
                'max_similarity': max_similarity if all_results else 0,
                'min_similarity': min_similarity if all_results else 0,
                'relevance_percentage': relevance_pct if all_results else 0
            }
        }
        
        report_file = f"vector_search_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{Colors.GREEN}Detailed report saved to: {report_file}{Colors.RESET}")
        
        # Final verdict
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}FINAL VERDICT{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        if all_results and relevance_pct >= 60:
            print(f"{Colors.GREEN}✓ VECTOR SEARCH IS WORKING PROPERLY{Colors.RESET}")
            print(f"  The system successfully returns relevant menopause-related content.")
            print(f"  {len(all_results)} total results across {len(test_queries)} test queries.")
        else:
            print(f"{Colors.RED}✗ VECTOR SEARCH NEEDS ATTENTION{Colors.RESET}")
            print(f"  Please review the results and consider re-ingesting documents.")

def main():
    """Main test function"""
    try:
        tester = VectorSearchTester()
        tester.test_search_queries()
        return 0
    except Exception as e:
        print(f"{Colors.RED}Test failed: {e}{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())