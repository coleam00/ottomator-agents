import pytest
import asyncio
import os
import json
import inspect
from unittest.mock import patch, MagicMock, AsyncMock

# Import the functions to test
from mcp_agent_army import get_model, select_model_for_task, MODEL_SELECTOR_DEFAULT, model_selection_agent


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


# Test the get_model function with different environment settings
def test_get_model_environment_override():
    """Test get_model behavior with different environment settings."""
    # Save original env values to restore later
    original_model = os.environ.get('MODEL_CHOICE')
    original_base_url = os.environ.get('BASE_URL')
    original_api_key = os.environ.get('LLM_API_KEY')
    
    try:
        # Test with custom environment settings
        os.environ['MODEL_CHOICE'] = 'gpt-4-turbo'
        os.environ['BASE_URL'] = 'https://custom-openai-endpoint.com/v1'
        os.environ['LLM_API_KEY'] = 'test-api-key'
        
        # Just test the model name since the provider structure is implementation-specific
        model = get_model()
        assert model._model_name == 'gpt-4-turbo'
        
        # Test with model override that should take precedence over env
        model = get_model('gpt-3.5-turbo')
        assert model._model_name == 'gpt-3.5-turbo'
    finally:
        # Restore original environment or remove it
        if original_model:
            os.environ['MODEL_CHOICE'] = original_model
        else:
            os.environ.pop('MODEL_CHOICE', None)
            
        if original_base_url:
            os.environ['BASE_URL'] = original_base_url
        else:
            os.environ.pop('BASE_URL', None)
            
        if original_api_key:
            os.environ['LLM_API_KEY'] = original_api_key
        else:
            os.environ.pop('LLM_API_KEY', None)


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


# Test the select_model_for_task with edge cases
@pytest.mark.asyncio
async def test_select_model_for_task_edge_cases():
    """Test select_model_for_task with various edge cases."""
    # Test with empty query
    mock_response = MockAgentResponse(json.dumps({
        "model": "gpt-3.5-turbo",
        "rephrased_query": "Please provide a valid question or query."
    }))
    
    with patch('mcp_agent_army.model_selection_agent.run', new=AsyncMock(return_value=mock_response)):
        model, rephrased = await select_model_for_task("")
        assert model == "gpt-3.5-turbo"
        assert rephrased == "Please provide a valid question or query."
    
    # Test with very short query
    mock_response = MockAgentResponse(json.dumps({
        "model": "gpt-3.5-turbo",
        "rephrased_query": "Hello?"
    }))
    
    with patch('mcp_agent_army.model_selection_agent.run', new=AsyncMock(return_value=mock_response)):
        model, rephrased = await select_model_for_task("hi")
        assert model == "gpt-3.5-turbo"
        assert rephrased == "Hello?"
    
    # Test with very long query (simulate a complex request)
    long_query = "Analyze in extensive detail the historical, economic, political, and cultural factors that have contributed to climate change over the past century, and provide a comprehensive framework for potential solutions at local, national, and international levels, considering technological innovations, policy reforms, behavioral changes, and ethical considerations." * 3
    
    mock_response = MockAgentResponse(json.dumps({
        "model": "gpt-4o",
        "rephrased_query": "Provide a comprehensive analysis of climate change causes and solutions across multiple domains."
    }))
    
    with patch('mcp_agent_army.model_selection_agent.run', new=AsyncMock(return_value=mock_response)):
        model, rephrased = await select_model_for_task(long_query)
        assert model == "gpt-4o"
        assert rephrased == "Provide a comprehensive analysis of climate change causes and solutions across multiple domains."


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
    # Check that the agent is using the correct model name
    assert model_selection_agent.model._model_name == MODEL_SELECTOR_DEFAULT


# Test for proper JSON formatting in response
@pytest.mark.asyncio
async def test_model_selection_json_response():
    """Test that the model selection agent produces a properly formatted JSON response."""
    # Create a mock for the run function that returns raw response text
    mock_run_result = MagicMock()
    mock_run_result._data = '{"model": "gpt-4o-mini", "rephrased_query": "Test query"}'
    
    # Check that parse_json_response works correctly
    response = MockAgentResponse(mock_run_result._data)
    parsed = response.parse_json_response()
    assert isinstance(parsed, dict)
    assert "model" in parsed
    assert "rephrased_query" in parsed
    assert parsed["model"] == "gpt-4o-mini"
    assert parsed["rephrased_query"] == "Test query"


# Test select_model_for_task instruction formatting
@pytest.mark.asyncio
async def test_select_model_instruction_format():
    """Test that the instruction string for model selection is formatted correctly."""
    # Spy on the function to capture the instruction
    original_agent_run = model_selection_agent.run
    instruction_captured = []
    
    async def mock_run(instruction):
        instruction_captured.append(instruction)
        mock_response = MagicMock()
        mock_response._data = '{"model": "gpt-3.5-turbo", "rephrased_query": "Test"}'
        mock_response.parse_json_response = lambda: json.loads(mock_response._data)
        return mock_response
    
    model_selection_agent.run = mock_run
    
    try:
        # Call the function
        await select_model_for_task("Test query")
        
        # Verify the instruction format
        assert len(instruction_captured) == 1
        instruction = instruction_captured[0]
        
        # Check instruction contains the expected components
        assert "USER QUERY: Test query" in instruction
        assert "gpt-3.5-turbo" in instruction
        assert "gpt-4o-mini" in instruction
        assert "gpt-4o" in instruction
        assert "select the most appropriate OpenAI model" in instruction
    finally:
        # Restore the original function
        model_selection_agent.run = original_agent_run


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


# Test for switching between different models
@pytest.mark.asyncio
async def test_model_switching():
    """Test switching between different models during processing."""
    # Mock primary agent
    mock_primary_agent = MagicMock()
    initial_model = get_model()
    mock_primary_agent.model = initial_model
    
    # Define a sequence of model switches
    model_sequence = [
        ("gpt-3.5-turbo", "Simple factual query"),
        ("gpt-4o-mini", "Medium complexity reasoning task"),
        ("gpt-4o", "Complex creative task"),
        (None, "Back to default model")
    ]
    
    # Mock the model selection function for each case
    for model_name, query in model_sequence:
        if model_name:
            # Update the model
            mock_primary_agent.model = get_model(model_name)
            # Verify model was updated correctly
            assert mock_primary_agent.model._model_name == model_name
        else:
            # Reset to default
            mock_primary_agent.model = get_model()
            assert mock_primary_agent.model._model_name == os.getenv('MODEL_CHOICE', 'gpt-4o-mini')


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 