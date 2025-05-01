import pytest
import asyncio
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Import the functions to test
from mcp_agent_army import get_model, select_model_for_task, MODEL_SELECTOR_DEFAULT


# Helper class to simulate the response from the model selection agent
class MockAgentResponse:
    def __init__(self, data):
        self._data = data

    def parse_json_response(self):
        return json.loads(self._data)


# Test the get_model function
def test_get_model_default():
    """Test get_model returns correct model with default settings."""
    # Save original env value to restore later
    original_model = os.environ.get('MODEL_CHOICE')
    
    try:
        # Set environment variable for test
        os.environ['MODEL_CHOICE'] = 'gpt-4o-mini'
        
        # Test default behavior
        model = get_model()
        # Check the model name passed to the OpenAIModel constructor
        assert model._model_name == 'gpt-4o-mini'
        
        # Test with explicit model name
        model = get_model('gpt-4o')
        assert model._model_name == 'gpt-4o'
    finally:
        # Restore original environment or remove it
        if original_model:
            os.environ['MODEL_CHOICE'] = original_model
        else:
            os.environ.pop('MODEL_CHOICE', None)


# Test the select_model_for_task function
@pytest.mark.asyncio
async def test_select_model_for_task():
    """Test that select_model_for_task correctly selects models based on query complexity."""
    
    # Define test cases
    test_cases = [
        {
            "query": "What is the weather today?",
            "expected_model": "gpt-3.5-turbo",
            "expected_rephrased": "What is the current weather forecast for today?"
        },
        {
            "query": "Write a detailed analysis of quantum computing applications in cybersecurity.",
            "expected_model": "gpt-4o",
            "expected_rephrased": "Provide a comprehensive analysis of how quantum computing technologies can be applied to enhance or threaten cybersecurity measures, including current research and future implications."
        },
        {
            "query": "Explain the concept of neural networks in machine learning.",
            "expected_model": "gpt-4o-mini",
            "expected_rephrased": "Provide a clear explanation of neural networks in machine learning, including their basic structure, how they work, and common applications."
        }
    ]
    
    for test_case in test_cases:
        # Set up mock response for the model selection agent
        mock_response = MockAgentResponse(json.dumps({
            "model": test_case["expected_model"],
            "rephrased_query": test_case["expected_rephrased"]
        }))
        
        # Patch the agent.run method to return our mock response
        with patch('mcp_agent_army.model_selection_agent.run', new=AsyncMock(return_value=mock_response)):
            # Call the function being tested
            model, rephrased = await select_model_for_task(test_case["query"])
            
            # Assert the results match expectations
            assert model == test_case["expected_model"]
            assert rephrased == test_case["expected_rephrased"]


# Test error handling in select_model_for_task
@pytest.mark.asyncio
async def test_select_model_for_task_error_handling():
    """Test that select_model_for_task handles errors gracefully."""
    
    # Set default model for the test
    original_model = os.environ.get('MODEL_CHOICE')
    os.environ['MODEL_CHOICE'] = 'gpt-4o-mini'
    
    try:
        # Make the agent.run method raise an exception
        with patch('mcp_agent_army.model_selection_agent.run', new=AsyncMock(side_effect=Exception("Test exception"))):
            # Call the function being tested
            model, rephrased = await select_model_for_task("Some query")
            
            # Assert the fallback behavior works correctly
            assert model == 'gpt-4o-mini'  # Should fall back to the default model
            assert rephrased == "Some query"  # Should return original query unchanged
    finally:
        # Restore original environment or remove it
        if original_model:
            os.environ['MODEL_CHOICE'] = original_model
        else:
            os.environ.pop('MODEL_CHOICE', None)


# Test the model selection agent's default model
def test_model_selector_default():
    """Test that the model selection agent uses the correct default model."""
    assert MODEL_SELECTOR_DEFAULT == "gpt-3.5-turbo"


# Integration test simulating the main loop behavior
@pytest.mark.asyncio
async def test_auto_model_selection_integration():
    """Test the auto model selection behavior in a simulated main loop."""
    # Mock primary agent
    mock_primary_agent = MagicMock()
    mock_primary_agent.model = get_model()
    
    # Mock the model selection function
    async def mock_select_model(*args, **kwargs):
        return "gpt-4o", "Optimized query"
    
    # Simulate main loop behavior with auto model selection enabled
    user_input = "Analyze the impact of climate change on global agriculture."
    auto_model_selection = True
    
    with patch('mcp_agent_army.select_model_for_task', new=mock_select_model):
        if auto_model_selection:
            selected_model, processed_input = await mock_select_model(user_input)
            # Update the primary agent's model
            mock_primary_agent.model = get_model(selected_model)
            
            # Verify model was updated correctly
            assert mock_primary_agent.model._model_name == "gpt-4o"
            assert processed_input == "Optimized query"
            
            # After processing, reset model to default
            mock_primary_agent.model = get_model()
            assert mock_primary_agent.model._model_name == os.getenv('MODEL_CHOICE', 'gpt-4o-mini')


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 