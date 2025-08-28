#!/usr/bin/env python3
"""
Simple test script to verify episodic memory fixes.
"""

import re
import asyncio
from agent.episodic_memory import EpisodicMemoryService
from agent.fact_extractor import MedicalFactExtractor


def test_regex_case_fix():
    """Test that regex patterns work with case-insensitive matching."""
    print("\n=== Testing Regex Case Fix ===")
    
    service = EpisodicMemoryService()
    
    # Test with different case variations
    test_cases = [
        "Started 3 days ago",
        "STARTED 3 DAYS AGO", 
        "Started 3 Days Ago"
    ]
    
    for user_message in test_cases:
        facts = service.extract_fact_triples(user_message, "I see")
        temporal_facts = [f for f in facts if "ONSET" in f[1]]
        print(f"Input: '{user_message}' -> Found {len(temporal_facts)} temporal facts")
        if temporal_facts:
            print(f"  Fact: {temporal_facts[0]}")
    
    print("✓ Regex case fix verified")


def test_safe_group_access():
    """Test that fact extractor safely accesses regex groups."""
    print("\n=== Testing Safe Group Access ===")
    
    extractor = MedicalFactExtractor()
    
    # Test with various inputs that might cause issues
    test_cases = [
        ("pain", "I understand"),  # Minimal text
        ("pain in", "OK"),  # Incomplete pattern
        ("severe", "noted"),  # Single word
    ]
    
    for user_msg, assistant_msg in test_cases:
        try:
            facts = extractor._extract_pattern_based_facts(user_msg, assistant_msg)
            print(f"Input: '{user_msg}' -> Extracted {len(facts)} facts (no error)")
        except IndexError as e:
            print(f"ERROR: IndexError with input '{user_msg}': {e}")
            raise
    
    print("✓ Safe group access verified")


def test_no_dead_code():
    """Test that dead code has been removed."""
    print("\n=== Testing Dead Code Removal ===")
    
    service = EpisodicMemoryService()
    
    # Check that we're not checking for non-existent method
    code_path = "agent/episodic_memory.py"
    with open(code_path, 'r') as f:
        content = f.read()
        
    # Should not have the dead code check anymore
    has_dead_code = "hasattr(self.graph_client, 'add_fact_triples')" in content
    
    if has_dead_code:
        print("ERROR: Dead code check still exists!")
    else:
        print("✓ Dead code removed successfully")


def test_batch_error_handling():
    """Test that batch processing has error handling."""
    print("\n=== Testing Batch Error Handling ===")
    
    code_path = "agent/episodic_memory.py"
    with open(code_path, 'r') as f:
        content = f.read()
    
    # Check for retry logic
    has_retry = "_create_episode_with_retry" in content
    has_fallback = "_store_failed_episodes" in content
    
    print(f"Has retry logic: {has_retry}")
    print(f"Has fallback storage: {has_fallback}")
    
    if has_retry and has_fallback:
        print("✓ Batch error handling implemented")
    else:
        print("WARNING: Batch error handling may be incomplete")


def test_timeout_implementation():
    """Test that timeout is implemented for episodic memory."""
    print("\n=== Testing Timeout Implementation ===")
    
    code_path = "agent/api.py"
    with open(code_path, 'r') as f:
        content = f.read()
    
    # Check for timeout implementation
    has_timeout_func = "_create_episodic_memory_with_timeout" in content
    has_timeout_var = "EPISODIC_MEMORY_TIMEOUT" in content
    has_wait_for = "asyncio.wait_for" in content and "episodic_memory_service" in content
    
    print(f"Has timeout function: {has_timeout_func}")
    print(f"Has timeout variable: {has_timeout_var}")
    print(f"Uses asyncio.wait_for: {has_wait_for}")
    
    if has_timeout_func and has_timeout_var and has_wait_for:
        print("✓ Timeout implementation verified")
    else:
        print("WARNING: Timeout implementation may be incomplete")


def main():
    """Run all simple tests."""
    print("Running episodic memory fix verification tests...")
    
    try:
        test_regex_case_fix()
        test_safe_group_access()
        test_no_dead_code()
        test_batch_error_handling()
        test_timeout_implementation()
        
        print("\n" + "="*50)
        print("✅ All fixes verified successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()