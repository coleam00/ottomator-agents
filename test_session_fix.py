#!/usr/bin/env python3
"""
Test script to verify session UUID handling is working correctly.
"""

import asyncio
import uuid
import logging
from agent.unified_db_utils import create_session, get_session
from agent.api import get_or_create_session
from agent.models import ChatRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_session_creation():
    """Test that session creation returns valid UUIDs."""
    print("\n=== Testing Session Creation ===")
    
    # Test 1: Create a new session
    print("\n1. Creating new session...")
    session_id = await create_session(
        user_id="test_user",
        metadata={"test": True}
    )
    print(f"   Created session: {session_id}")
    
    # Validate it's a UUID
    try:
        uuid.UUID(session_id)
        print(f"   ✅ Session ID is a valid UUID")
    except ValueError:
        print(f"   ❌ Session ID is NOT a valid UUID: {session_id}")
        return False
    
    # Test 2: Retrieve the session
    print("\n2. Retrieving session...")
    session = await get_session(session_id)
    if session:
        print(f"   ✅ Session retrieved successfully")
        print(f"   Session data: {session}")
    else:
        print(f"   ❌ Failed to retrieve session")
        return False
    
    # Test 3: Test get_or_create_session with valid UUID
    print("\n3. Testing get_or_create_session with valid UUID...")
    request = ChatRequest(
        message="Test message",
        session_id=session_id,
        user_id="test_user"
    )
    
    retrieved_id = await get_or_create_session(request)
    if retrieved_id == session_id:
        print(f"   ✅ Correctly retrieved existing session")
    else:
        print(f"   ❌ Got different session ID: {retrieved_id}")
        return False
    
    # Test 4: Test get_or_create_session with invalid session ID
    print("\n4. Testing get_or_create_session with invalid session ID...")
    invalid_request = ChatRequest(
        message="Test message",
        session_id="session-1756406301659",  # Invalid format
        user_id="test_user"
    )
    
    new_session_id = await get_or_create_session(invalid_request)
    print(f"   New session created: {new_session_id}")
    
    try:
        uuid.UUID(new_session_id)
        print(f"   ✅ New session ID is a valid UUID")
    except ValueError:
        print(f"   ❌ New session ID is NOT a valid UUID: {new_session_id}")
        return False
    
    # Test 5: Test get_or_create_session with None session ID
    print("\n5. Testing get_or_create_session with None session ID...")
    none_request = ChatRequest(
        message="Test message",
        session_id=None,
        user_id="test_user"
    )
    
    fresh_session_id = await get_or_create_session(none_request)
    print(f"   Fresh session created: {fresh_session_id}")
    
    try:
        uuid.UUID(fresh_session_id)
        print(f"   ✅ Fresh session ID is a valid UUID")
    except ValueError:
        print(f"   ❌ Fresh session ID is NOT a valid UUID: {fresh_session_id}")
        return False
    
    print("\n=== All tests passed! ===")
    return True


async def main():
    """Run tests."""
    try:
        # Initialize database
        from agent.unified_db_utils import initialize_database, close_database
        
        print("Initializing database...")
        await initialize_database()
        
        # Run tests
        success = await test_session_creation()
        
        # Clean up
        await close_database()
        
        if success:
            print("\n✅ Session UUID handling is working correctly!")
        else:
            print("\n❌ Session UUID handling has issues!")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())