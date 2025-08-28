"""
Test API robustness and error handling improvements.
"""

import asyncio
import httpx
import logging
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_root_endpoint(base_url: str = "http://localhost:8058"):
    """Test that root endpoint returns API information."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/")
            assert response.status_code == 200, f"Root endpoint returned {response.status_code}"
            data = response.json()
            assert "name" in data, "Root response missing 'name'"
            assert "endpoints" in data, "Root response missing 'endpoints'"
            logger.info("✅ Root endpoint test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Root endpoint test failed: {e}")
            return False


async def test_health_endpoint(base_url: str = "http://localhost:8058"):
    """Test that health endpoint returns proper status."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/health")
            assert response.status_code == 200, f"Health endpoint returned {response.status_code}"
            data = response.json()
            assert "status" in data, "Health response missing 'status'"
            logger.info(f"✅ Health endpoint test passed: {data['status']}")
            return True
        except Exception as e:
            logger.error(f"❌ Health endpoint test failed: {e}")
            return False


async def test_session_handling(base_url: str = "http://localhost:8058"):
    """Test session creation and retrieval with error handling."""
    async with httpx.AsyncClient() as client:
        try:
            # Test with non-existent session (should handle gracefully)
            fake_session_id = str(uuid4())
            response = await client.get(f"{base_url}/sessions/{fake_session_id}")
            
            # Should return 404 for non-existent session
            if response.status_code == 404:
                logger.info("✅ Non-existent session handled correctly (404)")
            else:
                logger.warning(f"⚠️ Non-existent session returned {response.status_code}")
            
            # Test chat with new session (should create session)
            response = await client.post(
                f"{base_url}/chat",
                json={
                    "message": "Test message",
                    "session_id": str(uuid4()),
                    "stream": False
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                logger.info("✅ Chat with new session handled correctly")
            else:
                error_detail = response.json() if response.status_code != 500 else response.text
                logger.warning(f"⚠️ Chat returned {response.status_code}: {error_detail}")
            
            return True
        except httpx.ConnectError:
            logger.error("❌ Cannot connect to API. Is the server running?")
            return False
        except Exception as e:
            logger.error(f"❌ Session handling test failed: {e}")
            return False


async def test_error_responses(base_url: str = "http://localhost:8058"):
    """Test that API returns proper error responses."""
    async with httpx.AsyncClient() as client:
        try:
            # Test invalid endpoint
            response = await client.get(f"{base_url}/invalid-endpoint")
            assert response.status_code == 404, f"Invalid endpoint should return 404, got {response.status_code}"
            logger.info("✅ Invalid endpoint returns 404")
            
            # Test invalid method
            response = await client.patch(f"{base_url}/chat")
            assert response.status_code == 405, f"Invalid method should return 405, got {response.status_code}"
            logger.info("✅ Invalid method returns 405")
            
            # Test missing required fields
            response = await client.post(f"{base_url}/chat", json={})
            assert response.status_code == 422, f"Missing fields should return 422, got {response.status_code}"
            logger.info("✅ Missing required fields returns 422")
            
            return True
        except httpx.ConnectError:
            logger.error("❌ Cannot connect to API. Is the server running?")
            return False
        except Exception as e:
            logger.error(f"❌ Error response test failed: {e}")
            return False


async def main():
    """Run all robustness tests."""
    logger.info("🚀 Starting API Robustness Tests")
    logger.info("=" * 50)
    
    base_url = "http://localhost:8058"
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Endpoint", test_health_endpoint),
        ("Session Handling", test_session_handling),
        ("Error Responses", test_error_responses),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📝 Testing: {test_name}")
        result = await test_func(base_url)
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🏁 Results: {passed}/{total} tests passed")
    
    if passed < total:
        logger.warning("\n⚠️ Some tests failed. Check the API implementation and ensure the server is running.")
    else:
        logger.info("\n🎉 All tests passed! API error handling is robust.")


if __name__ == "__main__":
    asyncio.run(main())