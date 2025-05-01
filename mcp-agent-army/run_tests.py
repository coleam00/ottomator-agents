#!/usr/bin/env python
"""
Test runner script for the MCP Agent Army.
Runs pytest with the appropriate configurations.
"""

import pytest
import sys
import os

def run_tests():
    """Run the test suite for the MCP Agent Army."""
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Run pytest
    print("Running tests for MCP Agent Army...")
    
    # Run the tests with verbose output
    exit_code = pytest.main(["-v", "tests"])
    
    # Report the result
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(run_tests()) 