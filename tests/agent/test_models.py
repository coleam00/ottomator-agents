"""
Tests for Pydantic models.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from agent.models import (
    ChatRequest,
    SearchRequest,
    DocumentMetadata,
    ChunkResult,
    GraphSearchResult,
    SearchResponse,
    ChatResponse,
    StreamDelta,
    Document,
    Chunk,
    Session,
    Message,
    AgentDependencies,
    IngestionConfig,
    IngestionResult,
    ErrorResponse,
    HealthStatus,
    MessageRole,
    SearchType
)


class TestRequestModels:
    """Test request models."""
    
    def test_chat_request_valid(self):
        """Test valid chat request."""
        request = ChatRequest(
            message="What are Google's AI initiatives?",
            session_id="test-session",
            user_id="test-user",
            search_type=SearchType.HYBRID
        )
        
        assert request.message == "What are Google's AI initiatives?"
        assert request.session_id == "test-session"
        assert request.user_id == "test-user"
        assert request.search_type == SearchType.HYBRID
        assert request.metadata == {}
    
    def test_chat_request_minimal(self):
        """Test minimal chat request."""
        request = ChatRequest(message="Hello")
        
        assert request.message == "Hello"
        assert request.session_id is None
        assert request.user_id is None
        assert request.search_type == SearchType.HYBRID
        assert request.metadata == {}
    
    def test_search_request_valid(self):
        """Test valid search request."""
        request = SearchRequest(
            query="Microsoft AI",
            search_type=SearchType.VECTOR,
            limit=20
        )
        
        assert request.query == "Microsoft AI"
        assert request.search_type == SearchType.VECTOR
        assert request.limit == 20
        assert request.filters == {}
    
    def test_search_request_limit_validation(self):
        """Test search request limit validation."""
        # Test minimum limit
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)
        
        # Test maximum limit
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=100)
        
        # Test valid limits
        request = SearchRequest(query="test", limit=1)
        assert request.limit == 1
        
        request = SearchRequest(query="test", limit=50)
        assert request.limit == 50


class TestResponseModels:
    """Test response models."""
    
    def test_document_metadata(self):
        """Test document metadata model."""
        now = datetime.now()
        metadata = DocumentMetadata(
            id="doc-123",
            title="Test Document",
            source="test.md",
            metadata={"topic": "AI"},
            created_at=now,
            updated_at=now,
            chunk_count=5
        )
        
        assert metadata.id == "doc-123"
        assert metadata.title == "Test Document"
        assert metadata.source == "test.md"
        assert metadata.metadata == {"topic": "AI"}
        assert metadata.chunk_count == 5
    
    def test_chunk_result(self):
        """Test chunk result model."""
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=0.85,
            metadata={"index": 0},
            document_title="Test Doc",
            document_source="test.md"
        )
        
        assert chunk.chunk_id == "chunk-123"
        assert chunk.document_id == "doc-123"
        assert chunk.content == "Test content"
        assert chunk.score == 0.85
        assert chunk.document_title == "Test Doc"
    
    def test_chunk_result_score_validation(self):
        """Test chunk result score validation."""
        # Test score clamping with validator
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=1.5,  # > 1.0, should be clamped to 1.0
            document_title="Test Doc",
            document_source="test.md"
        )
        assert chunk.score == 1.0
        
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=-0.5,  # < 0.0, should be clamped to 0.0
            document_title="Test Doc",
            document_source="test.md"
        )
        assert chunk.score == 0.0
        
        # Test valid score
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=0.85,  # Valid score
            document_title="Test Doc",
            document_source="test.md"
        )
        assert chunk.score == 0.85
    
    def test_graph_search_result(self):
        """Test graph search result model."""
        now = datetime.now()
        result = GraphSearchResult(
            fact="Google acquired DeepMind",
            uuid="test-uuid",
            valid_at=now.isoformat(),
            invalid_at=None,
            source_node_uuid="source-uuid"
        )
        
        assert result.fact == "Google acquired DeepMind"
        assert result.uuid == "test-uuid"
        assert result.valid_at == now.isoformat()
        assert result.invalid_at is None
        assert result.source_node_uuid == "source-uuid"
    
    def test_search_response(self):
        """Test search response model."""
        chunk = ChunkResult(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Test content",
            score=0.85,
            document_title="Test Doc",
            document_source="test.md"
        )
        
        response = SearchResponse(
            results=[chunk],
            total_results=1,
            search_type=SearchType.VECTOR,
            query_time_ms=150.5
        )
        
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.search_type == SearchType.VECTOR
        assert response.query_time_ms == 150.5
    
    def test_chat_response(self):
        """Test chat response model."""
        doc_metadata = DocumentMetadata(
            id="doc-123",
            title="Test Document",
            source="test.md",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        response = ChatResponse(
            message="Google is working on AI",
            session_id="session-123",
            sources=[doc_metadata],
            metadata={"tokens": 100}
        )
        
        assert response.message == "Google is working on AI"
        assert response.session_id == "session-123"
        assert len(response.sources) == 1
        assert response.metadata["tokens"] == 100


class TestDatabaseModels:
    """Test database models."""
    
    def test_document(self):
        """Test document model."""
        doc = Document(
            title="Test Document",
            source="test.md",
            content="Test content",
            metadata={"author": "Test"}
        )
        
        assert doc.title == "Test Document"
        assert doc.source == "test.md"
        assert doc.content == "Test content"
        assert doc.metadata == {"author": "Test"}
        assert doc.id is None  # Not set
    
    def test_chunk(self):
        """Test chunk model."""
        chunk = Chunk(
            document_id="doc-123",
            content="Test chunk content",
            embedding=[0.1, 0.2, 0.3] + [0.0] * 1533,  # 1536 dimensions
            chunk_index=0,
            metadata={"position": "start"},
            token_count=50
        )
        
        assert chunk.document_id == "doc-123"
        assert chunk.content == "Test chunk content"
        assert len(chunk.embedding) == 1536
        assert chunk.chunk_index == 0
        assert chunk.token_count == 50
    
    def test_chunk_embedding_validation(self):
        """Test chunk embedding dimension validation."""
        # Test wrong dimension
        with pytest.raises(ValueError, match="Embedding must have 1536 dimensions"):
            Chunk(
                document_id="doc-123",
                content="Test content",
                embedding=[0.1, 0.2],  # Wrong dimension
                chunk_index=0
            )
        
        # Test None embedding (should be valid)
        chunk = Chunk(
            document_id="doc-123",
            content="Test content",
            embedding=None,
            chunk_index=0
        )
        assert chunk.embedding is None
    
    def test_session(self):
        """Test session model."""
        now = datetime.now()
        session = Session(
            user_id="user-123",
            metadata={"client": "web"},
            created_at=now,
            expires_at=now
        )
        
        assert session.user_id == "user-123"
        assert session.metadata == {"client": "web"}
        assert session.created_at == now
        assert session.expires_at == now
    
    def test_message(self):
        """Test message model."""
        message = Message(
            session_id="session-123",
            role=MessageRole.USER,
            content="Hello",
            metadata={"client_ip": "127.0.0.1"}
        )
        
        assert message.session_id == "session-123"
        assert message.role == MessageRole.USER
        assert message.content == "Hello"
        assert message.metadata == {"client_ip": "127.0.0.1"}


class TestConfigurationModels:
    """Test configuration models."""
    
    def test_agent_dependencies(self):
        """Test agent dependencies model."""
        deps = AgentDependencies(
            session_id="session-123",
            database_url="postgresql://test",
            openai_api_key="sk-test"
        )
        
        assert deps.session_id == "session-123"
        assert deps.database_url == "postgresql://test"
        assert deps.openai_api_key == "sk-test"
    
    def test_ingestion_config(self):
        """Test ingestion configuration."""
        config = IngestionConfig(
            chunk_size=1000,
            chunk_overlap=200,
            max_chunk_size=2000,
            use_semantic_chunking=True,
            extract_entities=True
        )
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.use_semantic_chunking is True
        assert config.extract_entities is True
    
    def test_ingestion_config_validation(self):
        """Test ingestion config validation."""
        # Test invalid overlap (>= chunk_size)
        with pytest.raises(ValueError, match="Chunk overlap .* must be less than chunk size"):
            IngestionConfig(
                chunk_size=1000,
                chunk_overlap=1000  # Same as chunk_size
            )
        
        # Test valid configuration
        config = IngestionConfig(
            chunk_size=1000,
            chunk_overlap=200
        )
        assert config.chunk_overlap == 200
    
    def test_ingestion_result(self):
        """Test ingestion result model."""
        result = IngestionResult(
            document_id="doc-123",
            title="Test Document",
            chunks_created=10,
            entities_extracted=25,
            relationships_created=8,
            processing_time_ms=1500.0,
            errors=["Warning: Large document"]
        )
        
        assert result.document_id == "doc-123"
        assert result.title == "Test Document"
        assert result.chunks_created == 10
        assert result.entities_extracted == 25
        assert result.relationships_created == 8
        assert result.processing_time_ms == 1500.0
        assert len(result.errors) == 1


class TestUtilityModels:
    """Test utility models."""
    
    def test_stream_delta(self):
        """Test stream delta model."""
        delta = StreamDelta(
            content="Hello",
            delta_type="text",
            metadata={"position": 0}
        )
        
        assert delta.content == "Hello"
        assert delta.delta_type == "text"
        assert delta.metadata == {"position": 0}
    
    def test_error_response(self):
        """Test error response model."""
        error = ErrorResponse(
            error="Something went wrong",
            error_type="ValueError",
            details={"code": 400},
            request_id="req-123"
        )
        
        assert error.error == "Something went wrong"
        assert error.error_type == "ValueError"
        assert error.details == {"code": 400}
        assert error.request_id == "req-123"
    
    def test_health_status(self):
        """Test health status model."""
        now = datetime.now()
        health = HealthStatus(
            status="healthy",
            database=True,
            graph_database=True,
            llm_connection=True,
            version="0.1.0",
            timestamp=now
        )
        
        assert health.status == "healthy"
        assert health.database is True
        assert health.graph_database is True
        assert health.llm_connection is True
        assert health.version == "0.1.0"
        assert health.timestamp == now