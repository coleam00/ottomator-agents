"""
Gemini JSON extraction patch for Graphiti.

This module patches the Graphiti GeminiClient to properly handle
JSON responses that are wrapped in markdown code blocks.
"""

import re
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON from markdown code blocks.
    
    Gemini often returns JSON wrapped in markdown code blocks like:
    ```json
    {"key": "value"}
    ```
    
    This function extracts the JSON content from such blocks.
    
    Args:
        text: The raw text that may contain markdown-wrapped JSON
        
    Returns:
        The extracted JSON string, or the original text if no markdown block found
    """
    if not text:
        return text
    
    # Pattern to match markdown code blocks with optional language specifier
    # Matches: ```json {...} ``` or ``` {...} ```
    patterns = [
        r'```(?:json)?\s*\n?(.*?)\n?```',  # With or without 'json' specifier
        r'```(?:JSON)?\s*\n?(.*?)\n?```',  # Case variation
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.debug(f"Extracted JSON from markdown block: {extracted[:100]}...")
            return extracted
    
    # If no markdown blocks found, return original text
    return text


def patch_gemini_client_salvage_json():
    """
    Patch the GeminiClient's salvage_json method to handle markdown-wrapped JSON.
    
    This function modifies the graphiti_core.llm_client.gemini_client module
    to properly extract JSON from markdown code blocks before parsing.
    """
    try:
        # Import the module to patch
        from graphiti_core.llm_client import gemini_client
        
        # Check if salvage_json exists
        if not hasattr(gemini_client, 'salvage_json'):
            logger.warning("salvage_json method not found in gemini_client module")
            return False
        
        # Store original method
        original_salvage_json = gemini_client.salvage_json
        
        def patched_salvage_json(json_str: str) -> Optional[dict]:
            """
            Enhanced salvage_json that handles markdown-wrapped JSON.
            
            Args:
                json_str: The raw string that may contain JSON
                
            Returns:
                Parsed JSON dict if successful, None otherwise
            """
            if not json_str:
                return None
            
            # First, try to extract from markdown if present
            extracted = extract_json_from_markdown(json_str)
            
            # Try direct parsing first
            try:
                return json.loads(extracted)
            except (json.JSONDecodeError, ValueError):
                pass
            
            # If that fails, use the original salvage logic
            return original_salvage_json(extracted)
        
        # Replace the method
        gemini_client.salvage_json = patched_salvage_json
        
        logger.info("Successfully patched GeminiClient salvage_json method for markdown extraction")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import gemini_client module: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to patch GeminiClient: {e}")
        return False


def patch_gemini_client_parse_response():
    """
    Alternative patch that targets the parse_response or generate methods.
    
    This is a fallback if salvage_json doesn't exist or isn't the right target.
    """
    try:
        from graphiti_core.llm_client.gemini_client import GeminiClient
        
        # Store original generate method if it exists
        if hasattr(GeminiClient, 'generate'):
            original_generate = GeminiClient.generate
            
            async def patched_generate(self, *args, **kwargs):
                """
                Patched generate method that handles markdown-wrapped JSON.
                """
                result = await original_generate(self, *args, **kwargs)
                
                # If result is a string that might contain JSON, extract it
                if isinstance(result, str):
                    result = extract_json_from_markdown(result)
                
                return result
            
            # Replace the method
            GeminiClient.generate = patched_generate
            logger.info("Successfully patched GeminiClient generate method")
            return True
            
    except Exception as e:
        logger.error(f"Failed to patch GeminiClient generate: {e}")
        return False


def apply_gemini_json_patch():
    """
    Apply all necessary patches for Gemini JSON extraction.
    
    This function attempts multiple patch strategies to ensure
    JSON extraction works properly.
    
    Returns:
        bool: True if at least one patch was successful
    """
    success = False
    
    # Try salvage_json patch first (most direct)
    if patch_gemini_client_salvage_json():
        success = True
    
    # Also try generate method patch as backup
    if patch_gemini_client_parse_response():
        success = True
    
    if not success:
        logger.warning("No Gemini JSON patches could be applied")
    
    return success