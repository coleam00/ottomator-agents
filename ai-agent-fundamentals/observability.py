import os
from langfuse import get_client
from pydantic_ai.agent import Agent


def setup_observability():
    """Initialize LangFuse observability for Pydantic AI agents."""
    langfuse = get_client()

    if langfuse.auth_check():
        print("✓ LangFuse connected\n")
        Agent.instrument_all()
        return True
    else:
        print("✗ LangFuse authentication failed - continuing without observability\n")
        return False
