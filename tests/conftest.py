"""
Pytest configuration and fixtures.
"""

import pytest
import asyncio
import os
import tempfile
from typing import Generator, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Set test environment
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test_password")
# Flexible provider configuration for tests
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_BASE_URL", "https://api.openai.com/v1")
os.environ.setdefault("LLM_API_KEY", "sk-test-key-for-testing")
os.environ.setdefault("LLM_CHOICE", "gpt-4-turbo-preview")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
os.environ.setdefault("EMBEDDING_API_KEY", "sk-test-key-for-testing")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("INGESTION_LLM_CHOICE", "gpt-4o-mini")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_database_pool():
    """Mock database pool for testing."""
    with patch('agent.db_utils.db_pool') as mock_pool:
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        yield mock_pool


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client for testing."""
    with patch('agent.providers.get_embedding_client') as mock_get_client:
        mock_client = AsyncMock()
        
        # Mock embedding response
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
        
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_llm_model():
    """Mock LLM model for testing."""
    with patch('agent.providers.get_llm_model') as mock_get_model:
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_pydantic_agent():
    """Mock Pydantic AI agent for testing."""
    with patch('pydantic_ai.Agent') as mock_agent_class:
        mock_agent = AsyncMock()
        
        # Mock agent run response
        mock_result = Mock()
        mock_result.data = "Mocked agent response"
        mock_result.tool_calls.return_value = []
        mock_agent.run = AsyncMock(return_value=mock_result)
        
        # Mock agent iter for streaming
        mock_run_context = AsyncMock()
        mock_run_context.__aenter__ = AsyncMock(return_value=mock_run_context)
        mock_run_context.__aexit__ = AsyncMock(return_value=None)
        mock_agent.iter.return_value = mock_run_context
        
        mock_agent_class.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def mock_graphiti_client():
    """Mock Graphiti client for testing."""
    with patch('agent.graph_utils.GraphitiClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock search results
        mock_client.search.return_value = [
            {
                "fact": "Test fact",
                "episodes": [{"id": "ep1", "content": "test episode"}],
                "created_at": "2024-01-01T00:00:00Z",
                "valid_at": "2024-01-01T00:00:00Z",
                "uuid": "test-uuid"
            }
        ]
        
        # Mock entity relationships
        mock_client.get_related_entities.return_value = {
            "central_entity": "Google",
            "related_entities": ["DeepMind", "Alphabet"],
            "relationships": [{"from": "Google", "to": "DeepMind", "type": "owns"}],
            "depth": 2
        }
        
        # Mock statistics
        mock_client.get_graph_statistics.return_value = {
            "total_nodes": 100,
            "total_relationships": 50,
            "node_types": {"Entity": 80, "Fact": 20},
            "relationship_types": {"OWNS": 25, "PARTNERS_WITH": 25}
        }
        
        yield mock_client


@pytest.fixture
def temp_documents_dir():
    """Create temporary documents directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test documents
        test_docs = {
            "doc1.md": """# Document 1
            
This is the first test document.
It contains some basic content for testing.

## Section 1
Content in section 1.

## Section 2
Content in section 2.""",
            
            "doc2.md": """# Document 2

This is the second test document.
It has different content structure.

### Subsection A
Content in subsection A.

### Subsection B
Content in subsection B.""",
            
            "doc3.txt": """Document 3 (Text Format)

This document is in plain text format.
It should still be processed correctly.

Content paragraph 1.
Content paragraph 2."""
        }
        
        for filename, content in test_docs.items():
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write(content)
        
        yield temp_dir


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    from ingestion.chunker import DocumentChunk
    
    chunks = [
        DocumentChunk(
            content="This is the first chunk of content.",
            index=0,
            start_char=0,
            end_char=36,
            metadata={"title": "Test Doc", "topic": "AI"},
            token_count=8
        ),
        DocumentChunk(
            content="This is the second chunk with different content.",
            index=1,
            start_char=37,
            end_char=85,
            metadata={"title": "Test Doc", "topic": "AI"},
            token_count=10
        ),
        DocumentChunk(
            content="The third and final chunk completes the document.",
            index=2,
            start_char=86,
            end_char=135,
            metadata={"title": "Test Doc", "topic": "AI"},
            token_count=9
        )
    ]
    
    # Add mock embeddings
    for chunk in chunks:
        chunk.embedding = [0.1] * 1536
    
    return chunks


@pytest.fixture
def sample_documents():
    """Sample document metadata for testing."""
    from agent.models import DocumentMetadata
    from datetime import datetime
    
    now = datetime.now()
    
    return [
        DocumentMetadata(
            id="doc-1",
            title="AI Research Overview",
            source="ai_research.md",
            metadata={"author": "Dr. Smith", "year": 2024},
            created_at=now,
            updated_at=now,
            chunk_count=5
        ),
        DocumentMetadata(
            id="doc-2",
            title="Machine Learning Basics",
            source="ml_basics.md",
            metadata={"author": "Prof. Jones", "year": 2024},
            created_at=now,
            updated_at=now,
            chunk_count=8
        )
    ]


@pytest.fixture
def mock_vector_search_results():
    """Mock vector search results."""
    from agent.models import ChunkResult
    
    return [
        ChunkResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Google's AI research focuses on large language models.",
            score=0.95,
            metadata={"topic": "AI", "company": "Google"},
            document_title="AI Research Overview",
            document_source="ai_research.md"
        ),
        ChunkResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="DeepMind has made breakthroughs in protein folding.",
            score=0.87,
            metadata={"topic": "AI", "company": "DeepMind"},
            document_title="AI Research Overview",
            document_source="ai_research.md"
        )
    ]


@pytest.fixture
def mock_graph_search_results():
    """Mock graph search results."""
    from agent.models import GraphSearchResult
    from datetime import datetime
    
    now = datetime.now()
    
    return [
        GraphSearchResult(
            fact="Google acquired DeepMind in 2014",
            episodes=[
                {"id": "ep1", "content": "Acquisition announcement", "source": "news.md"}
            ],
            created_at=now,
            valid_at=now,
            uuid="fact-1"
        ),
        GraphSearchResult(
            fact="Microsoft partnered with OpenAI",
            episodes=[
                {"id": "ep2", "content": "Partnership details", "source": "partnership.md"}
            ],
            created_at=now,
            valid_at=now,
            uuid="fact-2"
        )
    ]


@pytest.fixture
def test_session_data():
    """Test session data."""
    return {
        "session_id": "test-session-123",
        "user_id": "test-user-456",
        "metadata": {"client": "test", "version": "1.0"}
    }


@pytest.fixture
def test_message_data():
    """Test message data."""
    return [
        {
            "id": "msg-1",
            "role": "user",
            "content": "What are Google's AI initiatives?",
            "metadata": {"timestamp": "2024-01-01T00:00:00Z"}
        },
        {
            "id": "msg-2",
            "role": "assistant",
            "content": "Google has several AI initiatives including...",
            "metadata": {"timestamp": "2024-01-01T00:01:00Z"}
        }
    ]


# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for all tests."""
    # Disable logging during tests to reduce noise
    import logging
    logging.disable(logging.CRITICAL)
    
    yield
    
    # Re-enable logging after tests
    logging.disable(logging.NOTSET)


# Async test helpers
def async_test(coro):
    """Helper to run async test functions."""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


# Mark for integration tests
pytestmark = pytest.mark.asyncio