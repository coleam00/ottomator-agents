#!/usr/bin/env python
"""
Script to clean up __pycache__ directories after running tests.
This helps maintain a clean project structure.
"""

import os
import shutil
import sys

def find_pycache_dirs(start_path='.'):
    """Find all __pycache__ directories from the given start path."""
    pycache_dirs = []
    
    for root, dirs, files in os.walk(start_path):
        if '__pycache__' in dirs:
            pycache_dirs.append(os.path.join(root, '__pycache__'))
    
    return pycache_dirs

def clean_pycache():
    """Remove all __pycache__ directories and .pyc files."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find all __pycache__ directories
    pycache_dirs = find_pycache_dirs(project_root)
    
    if not pycache_dirs:
        print("No __pycache__ directories found.")
        return 0
    
    # Print the directories that will be removed
    print(f"Found {len(pycache_dirs)} __pycache__ directories:")
    for d in pycache_dirs:
        print(f"  - {os.path.relpath(d, project_root)}")
    
    # Ask for confirmation
    try:
        response = input("\nDo you want to remove these directories? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    
    # Remove the directories
    for d in pycache_dirs:
        try:
            shutil.rmtree(d)
            print(f"Removed: {os.path.relpath(d, project_root)}")
        except Exception as e:
            print(f"Error removing {d}: {e}")
    
    print("\nCleanup completed!")
    return 0

if __name__ == "__main__":
    sys.exit(clean_pycache()) 