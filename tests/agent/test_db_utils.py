"""
Tests for database utilities.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from agent.db_utils import (
    DatabasePool,
    create_session,
    get_session,
    update_session,
    add_message,
    get_session_messages,
    get_document,
    list_documents,
    vector_search,
    hybrid_search,
    get_document_chunks,
    test_connection as db_test_connection
)


class TestDatabasePool:
    """Test database pool management."""
    
    def test_init_with_url(self):
        """Test initialization with database URL."""
        url = "postgresql://user:pass@host:5432/db"
        pool = DatabasePool(url)
        assert pool.database_url == url
    
    def test_init_without_url(self):
        """Test initialization without URL raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL environment variable not set"):
                DatabasePool()
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test pool initialization."""
        pool = DatabasePool("postgresql://test")
        
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_pool = Mock()
            mock_create_pool.return_value = mock_pool
            
            await pool.initialize()
            
            assert pool.pool == mock_pool
            mock_create_pool.assert_called_once_with(
                "postgresql://test",
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test pool closure."""
        pool = DatabasePool("postgresql://test")
        mock_pool = AsyncMock()
        pool.pool = mock_pool
        
        await pool.close()
        
        mock_pool.close.assert_called_once()
        assert pool.pool is None
    
    @pytest.mark.asyncio
    async def test_acquire_context_manager(self):
        """Test connection acquisition."""
        pool = DatabasePool("postgresql://test")
        
        mock_connection = Mock()
        
        # Create a mock that directly returns a context manager
        class MockContextManager:
            async def __aenter__(self):
                return mock_connection
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        mock_pool = Mock()
        mock_pool.acquire = Mock(return_value=MockContextManager())
        
        pool.pool = mock_pool
        
        async with pool.acquire() as conn:
            assert conn == mock_connection


class TestSessionManagement:
    """Test session management functions."""
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test session creation."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {"id": "session-123"}
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            session_id = await create_session(
                user_id="user-123",
                metadata={"client": "web"},
                timeout_minutes=30
            )
            
            assert session_id == "session-123"
            mock_conn.fetchrow.assert_called_once()
            
            # Check the SQL call
            call_args = mock_conn.fetchrow.call_args
            assert "INSERT INTO sessions" in call_args[0][0]
            assert call_args[0][1] == "user-123"  # user_id
            assert json.loads(call_args[0][2]) == {"client": "web"}  # metadata
    
    @pytest.mark.asyncio
    async def test_get_session_exists(self):
        """Test getting existing session."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_result = {
                "id": "session-123",
                "user_id": "user-123",
                "metadata": '{"client": "web"}',
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=1)
            }
            mock_conn.fetchrow.return_value = mock_result
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_pool.acquire.return_value = mock_context_manager
            
            session = await get_session("session-123")
            
            assert session is not None
            assert session["id"] == "session-123"
            assert session["user_id"] == "user-123"
            assert session["metadata"] == {"client": "web"}
    
    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        """Test getting non-existent session."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = None
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            session = await get_session("nonexistent")
            
            assert session is None
    
    @pytest.mark.asyncio
    async def test_update_session(self):
        """Test session update."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.execute.return_value = "UPDATE 1"  # PostgreSQL result for 1 row updated
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await update_session("session-123", {"new_key": "new_value"})
            
            assert result is True
            mock_conn.execute.assert_called_once()


class TestMessageManagement:
    """Test message management functions."""
    
    @pytest.mark.asyncio
    async def test_add_message(self):
        """Test adding message."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {"id": "message-123"}
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            message_id = await add_message(
                session_id="session-123",
                role="user",
                content="Hello",
                metadata={"client": "web"}
            )
            
            assert message_id == "message-123"
            mock_conn.fetchrow.assert_called_once()
            
            # Check the SQL call
            call_args = mock_conn.fetchrow.call_args
            assert "INSERT INTO messages" in call_args[0][0]
            assert call_args[0][2] == "user"  # role
            assert call_args[0][3] == "Hello"  # content
    
    @pytest.mark.asyncio
    async def test_get_session_messages(self):
        """Test getting session messages."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_messages = [
                {
                    "id": "msg-1",
                    "role": "user",
                    "content": "Hello",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc)
                },
                {
                    "id": "msg-2",
                    "role": "assistant",
                    "content": "Hi there!",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc)
                }
            ]
            mock_conn.fetch.return_value = mock_messages
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            messages = await get_session_messages("session-123", limit=10)
            
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            mock_conn.fetch.assert_called_once()


class TestDocumentManagement:
    """Test document management functions."""
    
    @pytest.mark.asyncio
    async def test_get_document(self):
        """Test getting document."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_result = {
                "id": "doc-123",
                "title": "Test Document",
                "source": "test.md",
                "content": "Test content",
                "metadata": '{"author": "test"}',
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            mock_conn.fetchrow.return_value = mock_result
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_pool.acquire.return_value = mock_context_manager
            
            document = await get_document("doc-123")
            
            assert document is not None
            assert document["id"] == "doc-123"
            assert document["title"] == "Test Document"
            assert document["metadata"] == {"author": "test"}
    
    @pytest.mark.asyncio
    async def test_list_documents(self):
        """Test listing documents."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "id": "doc-1",
                    "title": "Document 1",
                    "source": "doc1.md",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "chunk_count": 5
                },
                {
                    "id": "doc-2",
                    "title": "Document 2",
                    "source": "doc2.md",
                    "metadata": '{}',
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "chunk_count": 3
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            documents = await list_documents(limit=10, offset=0)
            
            assert len(documents) == 2
            assert documents[0]["title"] == "Document 1"
            assert documents[1]["title"] == "Document 2"


class TestVectorSearch:
    """Test vector search functions."""
    
    @pytest.mark.asyncio
    async def test_vector_search(self):
        """Test vector similarity search."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Test content 1",
                    "similarity": 0.95,
                    "metadata": '{}',
                    "document_title": "Test Doc",
                    "document_source": "test.md"
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            embedding = [0.1] * 1536  # Mock embedding
            results = await vector_search(embedding, limit=5)
            
            assert len(results) == 1
            assert results[0]["chunk_id"] == "chunk-1"
            assert results[0]["similarity"] == 0.95
            
            # Check that match_chunks function was called
            mock_conn.fetch.assert_called_once()
            call_args = mock_conn.fetch.call_args
            assert "match_chunks" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """Test hybrid search."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Test content",
                    "combined_score": 0.90,
                    "vector_similarity": 0.85,
                    "text_similarity": 0.70,
                    "metadata": '{}',
                    "document_title": "Test Doc",
                    "document_source": "test.md"
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            embedding = [0.1] * 1536
            results = await hybrid_search(
                embedding=embedding,
                query_text="test query",
                limit=5,
                text_weight=0.3
            )
            
            assert len(results) == 1
            assert results[0]["combined_score"] == 0.90
            assert results[0]["vector_similarity"] == 0.85
            assert results[0]["text_similarity"] == 0.70
    
    @pytest.mark.asyncio
    async def test_get_document_chunks(self):
        """Test getting document chunks."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_results = [
                {
                    "chunk_id": "chunk-1",
                    "content": "First chunk",
                    "chunk_index": 0,
                    "metadata": '{}'
                },
                {
                    "chunk_id": "chunk-2",
                    "content": "Second chunk",
                    "chunk_index": 1,
                    "metadata": '{}'
                }
            ]
            mock_conn.fetch.return_value = mock_results
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            chunks = await get_document_chunks("doc-123")
            
            assert len(chunks) == 2
            assert chunks[0]["chunk_index"] == 0
            assert chunks[1]["chunk_index"] == 1


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await db_test_connection()
            
            assert result is True
            mock_conn.fetchval.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test failed connection test."""
        with patch('agent.db_utils.db_pool') as mock_pool:
            mock_pool.acquire.side_effect = Exception("Connection failed")
            
            result = await db_test_connection()
            
            assert result is False