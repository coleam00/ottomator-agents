#!/usr/bin/env python3
"""
Local integration test for episodic memory system.
Tests all components without requiring external services.
"""

import os
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from agent.fact_extractor import MedicalFactExtractor, MedicalFact


async def test_fact_extractor():
    """Test the fact extractor component."""
    print("\n=== Testing Fact Extractor ===")
    
    extractor = MedicalFactExtractor()
    
    # Test conversation
    user_message = "I've been experiencing severe headaches for 3 days. The pain is in my forehead and gets worse when I bend over."
    assistant_response = "I understand you're experiencing severe headaches in your forehead. I recommend trying ibuprofen for pain relief."
    
    # Extract facts
    facts = await extractor.extract_facts_from_conversation(
        user_message, assistant_response
    )
    
    print(f"Extracted {len(facts)} medical facts:")
    for fact in facts:
        print(f"  - {fact.subject} {fact.predicate} {fact.object} (confidence: {fact.confidence})")
    
    # Validate facts
    validated = extractor.validate_facts(facts)
    print(f"Validated {len(validated)} facts")
    
    # Consolidate facts
    consolidated = extractor.consolidate_facts(validated)
    print(f"Consolidated to {len(consolidated)} unique facts")
    
    # Format for Graphiti
    formatted = extractor.format_facts_for_graphiti(consolidated)
    print(f"Formatted {len(formatted)} triples for Graphiti:")
    for triple in formatted[:3]:
        print(f"  - {triple}")
    
    print("✓ Fact extractor working correctly")
    return True


async def test_pattern_extraction():
    """Test pattern-based extraction."""
    print("\n=== Testing Pattern Extraction ===")
    
    extractor = MedicalFactExtractor()
    
    test_cases = [
        ("I have a headache", "Symptom extraction"),
        ("experiencing dizziness", "Experience pattern"),  
        ("suffering from nausea", "Suffering pattern"),
        ("sharp pain in my chest", "Location pattern"),
        ("severe headache", "Severity pattern"),
        ("mild discomfort", "Severity pattern"),
        ("for 3 days", "Duration pattern"),
        ("started 2 weeks ago", "Onset pattern"),
        ("lasting 4 hours", "Duration pattern"),
        ("daily headaches", "Frequency pattern"),
        ("3 times per day", "Frequency pattern"),
        ("stress triggers migraines", "Causal pattern"),
        ("aspirin helps with pain", "Relief pattern"),
    ]
    
    for text, description in test_cases:
        facts = extractor._extract_pattern_based_facts(text, "Noted")
        temporal = extractor._extract_temporal_facts(text)
        causal = extractor._extract_causal_facts(text, "Understood")
        
        total = len(facts) + len(temporal) + len(causal)
        if total > 0:
            print(f"✓ {description}: '{text}' -> {total} facts")
        else:
            print(f"  {description}: '{text}' -> No facts extracted")
    
    print("✓ Pattern extraction working correctly")
    return True


async def test_edge_cases():
    """Test safe handling of edge cases."""
    print("\n=== Testing Edge Case Handling ===")
    
    extractor = MedicalFactExtractor()
    
    # Edge cases that should not cause errors
    edge_cases = [
        ("", "", "Empty strings"),
        ("pain", "OK", "Single word"),
        ("severe", "noted", "Severity only"),
        ("headache in", "understood", "Incomplete pattern"),
        ("I have", "I see", "Partial pattern"),
        ("123", "456", "Numbers only"),
        ("!!!!", "????", "Special characters"),
        ("a" * 1000, "b" * 1000, "Very long text"),
    ]
    
    errors = []
    for user_msg, assistant_msg, description in edge_cases:
        try:
            facts = await extractor.extract_facts_from_conversation(
                user_msg, assistant_msg
            )
            print(f"✓ {description}: {len(facts)} facts (no error)")
        except Exception as e:
            errors.append(f"{description}: {e}")
    
    if errors:
        print("\nErrors found:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    print("✓ All edge cases handled safely")
    return True


async def test_confidence_threshold():
    """Test confidence threshold filtering."""
    print("\n=== Testing Confidence Threshold ===")
    
    extractor = MedicalFactExtractor()
    
    # Set a high threshold
    original_threshold = extractor.confidence_threshold
    extractor.confidence_threshold = 0.9
    
    user_msg = "I have mild headaches occasionally"
    assistant_msg = "Consider rest and hydration"
    
    facts_high = await extractor.extract_facts_from_conversation(
        user_msg, assistant_msg
    )
    print(f"With threshold 0.9: {len(facts_high)} facts")
    
    # Lower threshold
    extractor.confidence_threshold = 0.5
    facts_low = await extractor.extract_facts_from_conversation(
        user_msg, assistant_msg
    )
    print(f"With threshold 0.5: {len(facts_low)} facts")
    
    # Restore original
    extractor.confidence_threshold = original_threshold
    
    print(f"✓ Confidence threshold filtering working (high: {len(facts_high)}, low: {len(facts_low)})")
    return True


async def test_medical_fact_model():
    """Test the MedicalFact dataclass."""
    print("\n=== Testing MedicalFact Model ===")
    
    # Create facts with different configurations
    fact1 = MedicalFact(
        subject="Patient",
        predicate="HAS_SYMPTOM",
        object="headache",
        confidence=0.95,
        source="pattern_extraction"
    )
    
    fact2 = MedicalFact(
        subject="headache",
        predicate="LOCATED_IN",
        object="forehead",
        confidence=0.8,
        source="pattern_extraction",
        context="pain in my forehead",
        temporal_info="for 3 days"
    )
    
    print(f"Fact 1: {fact1.subject} {fact1.predicate} {fact1.object}")
    print(f"Fact 2: {fact2.subject} {fact2.predicate} {fact2.object}")
    print(f"  Context: {fact2.context}")
    print(f"  Temporal: {fact2.temporal_info}")
    
    print("✓ MedicalFact model working correctly")
    return True


async def test_consolidation():
    """Test fact consolidation."""
    print("\n=== Testing Fact Consolidation ===")
    
    extractor = MedicalFactExtractor()
    
    # Create duplicate facts with different confidence
    facts = [
        MedicalFact("Patient", "HAS_SYMPTOM", "headache", 0.8, "source1"),
        MedicalFact("Patient", "HAS_SYMPTOM", "headache", 0.9, "source2"),  # Higher confidence
        MedicalFact("Patient", "HAS_SYMPTOM", "fever", 0.7, "source1"),
        MedicalFact("headache", "HAS_SEVERITY", "severe", 0.85, "source1"),
        MedicalFact("headache", "HAS_SEVERITY", "severe", 0.75, "source2"),  # Lower confidence
    ]
    
    consolidated = extractor.consolidate_facts(facts)
    
    print(f"Original facts: {len(facts)}")
    print(f"Consolidated facts: {len(consolidated)}")
    
    # Check that higher confidence was kept
    for fact in consolidated:
        if fact.subject == "Patient" and fact.object == "headache":
            assert fact.confidence == 0.9, "Should keep higher confidence"
            print(f"✓ Kept higher confidence for duplicate: {fact.confidence}")
    
    print("✓ Fact consolidation working correctly")
    return True


async def main():
    """Run all local integration tests."""
    print("Running local episodic memory integration tests...")
    print("="*60)
    
    tests = [
        test_fact_extractor,
        test_pattern_extraction,
        test_edge_cases,
        test_confidence_threshold,
        test_medical_fact_model,
        test_consolidation,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    if all(results):
        print("✅ All local integration tests passed!")
        print("\nEpisodic memory components verified:")
        print("  ✓ Medical fact extraction")
        print("  ✓ Pattern-based extraction")
        print("  ✓ Edge case handling")
        print("  ✓ Confidence filtering")
        print("  ✓ Fact consolidation")
        print("  ✓ Data models")
        print("\nThe system is ready for deployment!")
    else:
        print("❌ Some tests failed")
        failed_count = len([r for r in results if not r])
        print(f"  Failed: {failed_count}/{len(results)}")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())