import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import os
import json
import sys

# Add try/except block to handle the case where the studio integration module might not be found
try:
    # Try to import the FastAPI app from the studio integration version
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fastapi.testclient import TestClient
    from studio_integration_version.mcp_agent_army_endpoint import app, AgentRequest
    STUDIO_INTEGRATION_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    # Skip tests if the module is not available
    STUDIO_INTEGRATION_AVAILABLE = False
    pass

# Skip all tests if the studio integration version is not available
pytestmark = pytest.mark.skipif(
    not STUDIO_INTEGRATION_AVAILABLE,
    reason="Studio integration version not available"
)

# Setup test client
@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    if not STUDIO_INTEGRATION_AVAILABLE:
        pytest.skip("Studio integration version not available")
    return TestClient(app)


# Mock the dependencies to avoid actual API calls and database access
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """Mock dependencies to isolate the API endpoints for testing."""
    if not STUDIO_INTEGRATION_AVAILABLE:
        pytest.skip("Studio integration version not available")
        
    # Mock the primary agent
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=MagicMock(data="Test response"))
    
    # Mock supabase client
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_supabase
    mock_supabase.select.return_value = mock_supabase
    mock_supabase.eq.return_value = mock_supabase
    mock_supabase.order.return_value = mock_supabase
    mock_supabase.limit.return_value = mock_supabase
    mock_supabase.insert.return_value = mock_supabase
    mock_supabase.execute.return_value = MagicMock(data=[])
    
    # Mock the select_model_for_task function
    async def mock_select_model(*args, **kwargs):
        return "gpt-4o", "Optimized test query"
    
    # Mock get_model function
    def mock_get_model(model_name=None):
        model = MagicMock()
        model.model = model_name or os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
        return model
    
    # Apply the mocks
    monkeypatch.setattr("studio_integration_version.mcp_agent_army_endpoint.primary_agent", mock_agent)
    monkeypatch.setattr("studio_integration_version.mcp_agent_army_endpoint.supabase", mock_supabase)
    monkeypatch.setattr("studio_integration_version.mcp_agent_army_endpoint.select_model_for_task", mock_select_model)
    monkeypatch.setattr("studio_integration_version.mcp_agent_army_endpoint.get_model", mock_get_model)
    
    # Mock API token verification
    monkeypatch.setattr("studio_integration_version.mcp_agent_army_endpoint.verify_token", lambda x: True)
    
    # Set API_BEARER_TOKEN environment variable for testing
    monkeypatch.setenv("API_BEARER_TOKEN", "test-token")


# Test the main endpoint with auto model selection enabled
def test_mcp_agent_army_with_auto_model(client):
    """Test the /api/mcp-agent-army endpoint with auto model selection enabled."""
    # Create test request
    request_data = {
        "query": "What is the weather today?",
        "user_id": "test-user",
        "request_id": "test-request",
        "session_id": "test-session",
        "auto_model_selection": True
    }
    
    # Set up headers
    headers = {
        "Authorization": "Bearer test-token"
    }
    
    # Make the request
    response = client.post("/api/mcp-agent-army", json=request_data, headers=headers)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_used"] == "gpt-4o"
    assert data["rephrased_query"] == "Optimized test query"


# Test the main endpoint with auto model selection disabled
def test_mcp_agent_army_without_auto_model(client):
    """Test the /api/mcp-agent-army endpoint with auto model selection disabled."""
    # Create test request
    request_data = {
        "query": "What is the weather today?",
        "user_id": "test-user",
        "request_id": "test-request",
        "session_id": "test-session",
        "auto_model_selection": False
    }
    
    # Set up headers
    headers = {
        "Authorization": "Bearer test-token"
    }
    
    # Make the request
    response = client.post("/api/mcp-agent-army", json=request_data, headers=headers)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_used"] is None
    assert data["rephrased_query"] is None


# Test the toggle endpoint
def test_toggle_model_selection(client):
    """Test the /api/toggle-model-selection endpoint."""
    # Set up headers
    headers = {
        "Authorization": "Bearer test-token"
    }
    
    # Test enabling auto model selection
    params = {
        "session_id": "test-session",
        "enable": True
    }
    
    response = client.post("/api/toggle-model-selection", params=params, headers=headers)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["auto_model_selection"] is True
    
    # Test disabling auto model selection
    params = {
        "session_id": "test-session",
        "enable": False
    }
    
    response = client.post("/api/toggle-model-selection", params=params, headers=headers)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["auto_model_selection"] is False


# Test error handling in the main endpoint
def test_mcp_agent_army_error_handling(client, monkeypatch):
    """Test error handling in the /api/mcp-agent-army endpoint."""
    # Mock primary agent to raise an exception
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=Exception("Test exception"))
    monkeypatch.setattr("studio_integration_version.mcp_agent_army_endpoint.primary_agent", mock_agent)
    
    # Create test request
    request_data = {
        "query": "What is the weather today?",
        "user_id": "test-user",
        "request_id": "test-request",
        "session_id": "test-session"
    }
    
    # Set up headers
    headers = {
        "Authorization": "Bearer test-token"
    }
    
    # Make the request
    response = client.post("/api/mcp-agent-army", json=request_data, headers=headers)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False


if __name__ == "__main__":
    if STUDIO_INTEGRATION_AVAILABLE:
        pytest.main(["-xvs", __file__])
    else:
        print("Studio integration version not available. Skipping API tests.") 