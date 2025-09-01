#!/usr/bin/env python3
"""
Integration test for the complete RAG system with dual-search architecture.
Tests session management, vector search, knowledge base search, and episodic memory.
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8058"


async def test_health_check():
    """Test API health check."""
    print("\n=== Testing Health Check ===")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Health Status: {data.get('status')}")
                print(f"   Database: {data.get('database')}")
                print(f"   Graph: {data.get('graph_database')}")
                print(f"   Provider: {data.get('provider')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status}")
                return False


async def test_vector_search():
    """Test vector search functionality."""
    print("\n=== Testing Vector Search ===")
    async with aiohttp.ClientSession() as session:
        query_data = {
            "query": "What are common menopause symptoms?",
            "limit": 5
        }
        
        async with session.post(
            f"{BASE_URL}/search/vector",
            json=query_data
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Vector search returned {data.get('total_results')} results")
                print(f"   Query time: {data.get('query_time_ms'):.2f}ms")
                
                # Show first result
                if data.get('results'):
                    first = data['results'][0]
                    print(f"   Top result (score={first.get('score', 0):.3f}):")
                    print(f"   {first.get('content', '')[:200]}...")
                return True
            else:
                print(f"❌ Vector search failed: {response.status}")
                error = await response.text()
                print(f"   Error: {error}")
                return False


async def test_knowledge_base_search():
    """Test knowledge base search functionality."""
    print("\n=== Testing Knowledge Base Search ===")
    async with aiohttp.ClientSession() as session:
        query_data = {
            "query": "menopause treatments",
            "limit": 5
        }
        
        async with session.post(
            f"{BASE_URL}/search/graph",
            json=query_data
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Knowledge base search returned {data.get('total_results')} results")
                print(f"   Query time: {data.get('query_time_ms'):.2f}ms")
                
                # Show first result
                if data.get('graph_results'):
                    first = data['graph_results'][0]
                    print(f"   Top result:")
                    print(f"   {first.get('fact', '')[:200]}...")
                return True
            else:
                print(f"❌ Knowledge base search failed: {response.status}")
                error = await response.text()
                print(f"   Error: {error}")
                return False


async def test_chat_with_session():
    """Test chat functionality with proper session management."""
    print("\n=== Testing Chat with Session Management ===")
    
    async with aiohttp.ClientSession() as session:
        # First chat message - should create a new session
        chat_data = {
            "message": "What are the most common symptoms of menopause?",
            "user_id": "test_user_integration"
        }
        
        print("\n1. Sending first message (creating session)...")
        async with session.post(
            f"{BASE_URL}/chat",
            json=chat_data
        ) as response:
            if response.status != 200:
                print(f"❌ First chat failed: {response.status}")
                error = await response.text()
                print(f"   Error: {error}")
                return False
            
            data = await response.json()
            session_id = data.get('session_id')
            
            if not session_id:
                print(f"❌ No session ID returned")
                return False
            
            print(f"✅ Session created: {session_id}")
            print(f"   Response: {data.get('message', '')[:200]}...")
            
            # Check tools used
            tools = data.get('tools_used', [])
            if tools:
                print(f"   Tools used: {[t['tool_name'] for t in tools]}")
        
        # Second chat message - should use existing session
        chat_data2 = {
            "message": "Can you tell me more about hot flashes specifically?",
            "session_id": session_id,
            "user_id": "test_user_integration"
        }
        
        print("\n2. Sending follow-up message (using existing session)...")
        async with session.post(
            f"{BASE_URL}/chat",
            json=chat_data2
        ) as response:
            if response.status != 200:
                print(f"❌ Second chat failed: {response.status}")
                error = await response.text()
                print(f"   Error: {error}")
                return False
            
            data = await response.json()
            returned_session_id = data.get('session_id')
            
            if returned_session_id != session_id:
                print(f"❌ Session ID mismatch: expected {session_id}, got {returned_session_id}")
                return False
            
            print(f"✅ Used existing session: {returned_session_id}")
            print(f"   Response: {data.get('message', '')[:200]}...")
            
            # Check tools used
            tools = data.get('tools_used', [])
            if tools:
                print(f"   Tools used: {[t['tool_name'] for t in tools]}")
        
        # Verify session info
        print("\n3. Verifying session information...")
        async with session.get(
            f"{BASE_URL}/sessions/{session_id}"
        ) as response:
            if response.status != 200:
                print(f"❌ Failed to get session info: {response.status}")
                return False
            
            session_data = await response.json()
            print(f"✅ Session verified:")
            print(f"   User ID: {session_data.get('user_id')}")
            print(f"   Created: {session_data.get('created_at')}")
            print(f"   Expires: {session_data.get('expires_at')}")
    
    return True


async def test_streaming_chat():
    """Test streaming chat functionality."""
    print("\n=== Testing Streaming Chat ===")
    
    async with aiohttp.ClientSession() as session:
        chat_data = {
            "message": "What lifestyle changes can help with menopause symptoms?",
            "user_id": "test_user_streaming"
        }
        
        print("Sending streaming request...")
        async with session.post(
            f"{BASE_URL}/chat/stream",
            json=chat_data
        ) as response:
            if response.status != 200:
                print(f"❌ Streaming chat failed: {response.status}")
                error = await response.text()
                print(f"   Error: {error}")
                return False
            
            session_id = None
            tools_used = []
            full_response = ""
            
            print("Receiving stream...")
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        
                        if data.get('type') == 'session':
                            session_id = data.get('session_id')
                            print(f"✅ Session: {session_id}")
                        
                        elif data.get('type') == 'text':
                            content = data.get('content', '')
                            full_response += content
                            # Print dots to show progress
                            if len(full_response) % 100 == 0:
                                print(".", end="", flush=True)
                        
                        elif data.get('type') == 'tools':
                            tools_used = data.get('tools', [])
                        
                        elif data.get('type') == 'end':
                            print("\n✅ Stream completed")
                            break
                        
                        elif data.get('type') == 'error':
                            print(f"\n❌ Stream error: {data.get('content')}")
                            return False
                    
                    except json.JSONDecodeError:
                        continue
            
            print(f"   Response length: {len(full_response)} chars")
            if tools_used:
                print(f"   Tools used: {[t['tool_name'] for t in tools_used]}")
            
            return True


async def test_hybrid_search():
    """Test hybrid search functionality."""
    print("\n=== Testing Hybrid Search ===")
    async with aiohttp.ClientSession() as session:
        query_data = {
            "query": "hormone replacement therapy HRT",
            "limit": 5
        }
        
        async with session.post(
            f"{BASE_URL}/search/hybrid",
            json=query_data
        ) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Hybrid search returned {data.get('total_results')} results")
                print(f"   Query time: {data.get('query_time_ms'):.2f}ms")
                
                # Show first result
                if data.get('results'):
                    first = data['results'][0]
                    print(f"   Top result (score={first.get('score', 0):.3f}):")
                    print(f"   {first.get('content', '')[:200]}...")
                return True
            else:
                print(f"❌ Hybrid search failed: {response.status}")
                error = await response.text()
                print(f"   Error: {error}")
                return False


async def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("🧪 RUNNING INTEGRATION TESTS")
    print("="*60)
    
    # Give API time to start if just launched
    await asyncio.sleep(1)
    
    tests = [
        ("Health Check", test_health_check),
        ("Vector Search", test_vector_search),
        ("Knowledge Base Search", test_knowledge_base_search),
        ("Hybrid Search", test_hybrid_search),
        ("Chat with Session", test_chat_with_session),
        ("Streaming Chat", test_streaming_chat),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())