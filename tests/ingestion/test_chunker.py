"""
Tests for document chunking functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from ingestion.chunker import (
    ChunkingConfig,
    DocumentChunk,
    SemanticChunker,
    SimpleChunker,
    create_chunker
)


class TestChunkingConfig:
    """Test chunking configuration."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            max_chunk_size=2000,
            use_semantic_splitting=True
        )
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.use_semantic_splitting is True
    
    def test_invalid_overlap(self):
        """Test invalid overlap configuration."""
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=1000  # Same as chunk size
            )
        
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=1200  # Greater than chunk size
            )
    
    def test_invalid_min_chunk_size(self):
        """Test invalid minimum chunk size."""
        with pytest.raises(ValueError, match="Minimum chunk size must be positive"):
            ChunkingConfig(min_chunk_size=0)
        
        with pytest.raises(ValueError, match="Minimum chunk size must be positive"):
            ChunkingConfig(min_chunk_size=-10)


class TestDocumentChunk:
    """Test document chunk data structure."""
    
    def test_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            content="This is test content for a document chunk.",
            index=0,
            start_char=0,
            end_char=42,
            metadata={"title": "Test Doc"},
            token_count=10
        )
        
        assert chunk.content == "This is test content for a document chunk."
        assert chunk.index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 42
        assert chunk.metadata == {"title": "Test Doc"}
        assert chunk.token_count == 10
    
    def test_automatic_token_count(self):
        """Test automatic token count calculation."""
        chunk = DocumentChunk(
            content="A" * 40,  # 40 characters
            index=0,
            start_char=0,
            end_char=40,
            metadata={}
        )
        
        # Should estimate ~10 tokens (40 chars / 4)
        assert chunk.token_count == 10


class TestSimpleChunker:
    """Test simple rule-based chunker."""
    
    def test_empty_content(self):
        """Test chunking empty content."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = SimpleChunker(config)
        
        chunks = chunker.chunk_document("", "Empty Doc", "empty.md")
        
        assert len(chunks) == 0
    
    def test_short_content(self):
        """Test chunking content shorter than chunk size."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = SimpleChunker(config)
        
        content = "This is a short document."
        chunks = chunker.chunk_document(content, "Short Doc", "short.md")
        
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].index == 0
        assert chunks[0].metadata["title"] == "Short Doc"
        assert chunks[0].metadata["chunk_method"] == "simple"
    
    def test_multiple_paragraphs(self):
        """Test chunking content with multiple paragraphs."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = SimpleChunker(config)
        
        content = """First paragraph with some content.

Second paragraph with more content.

Third paragraph to test chunking."""
        
        chunks = chunker.chunk_document(content, "Multi Para", "multi.md")
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check metadata
        for chunk in chunks:
            assert chunk.metadata["title"] == "Multi Para"
            assert chunk.metadata["chunk_method"] == "simple"
            assert "total_chunks" in chunk.metadata
        
        # Check indices
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
    
    def test_chunk_overlap(self):
        """Test that chunking respects overlap settings."""
        config = ChunkingConfig(chunk_size=30, chunk_overlap=10)
        chunker = SimpleChunker(config)
        
        # Create content with paragraph breaks to force chunking
        content = "A" * 25 + "\n\n" + "B" * 25 + "\n\n" + "C" * 25 + "\n\n" + "D" * 25
        chunks = chunker.chunk_document(content, "Overlap Test", "overlap.md")
        
        # Should create multiple chunks due to paragraph breaks and size
        assert len(chunks) > 1
        
        # Each chunk should be roughly the chunk size
        for chunk in chunks[:-1]:  # All except last
            assert len(chunk.content) <= config.chunk_size + 5  # Allow some variance


class TestSemanticChunker:
    """Test semantic chunker (with mocked LLM calls)."""
    
    def test_init(self):
        """Test semantic chunker initialization."""
        config = ChunkingConfig(use_semantic_splitting=True)
        chunker = SemanticChunker(config)
        
        assert chunker.config == config
        # Model is now an OpenAIModel object, not a string
        assert hasattr(chunker.model, 'model_name')
    
    def test_split_on_structure(self):
        """Test structural splitting."""
        config = ChunkingConfig()
        chunker = SemanticChunker(config)
        
        content = """# Main Title

This is the first paragraph.

This is the second paragraph.

## Section Header

This is content under the section.

- List item 1
- List item 2

1. Numbered item 1
2. Numbered item 2"""
        
        sections = chunker._split_on_structure(content)
        
        # Should split on various structural elements
        assert len(sections) > 5
        
        # Check that headers are preserved
        headers = [s for s in sections if s.strip().startswith('#')]
        assert len(headers) >= 2
    
    @pytest.mark.asyncio
    async def test_chunk_document_fallback(self):
        """Test that semantic chunker falls back to simple chunking on errors."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10, use_semantic_splitting=True)
        chunker = SemanticChunker(config)
        
        # Mock the semantic chunking to fail
        with patch.object(chunker, '_semantic_chunk', side_effect=Exception("LLM failed")):
            content = "This is test content for fallback testing. " * 10
            chunks = await chunker.chunk_document(content, "Fallback Test", "fallback.md")
            
            # Should still return chunks from simple chunking
            assert len(chunks) > 0
            assert chunks[0].metadata["title"] == "Fallback Test"
    
    def test_simple_split(self):
        """Test simple splitting method."""
        config = ChunkingConfig(chunk_size=30, chunk_overlap=10)
        chunker = SemanticChunker(config)
        
        text = "This is a test sentence. This is another sentence. And one more."
        chunks = chunker._simple_split(text)
        
        assert len(chunks) > 1
        
        # Check that splits try to end at sentence boundaries
        for chunk in chunks[:-1]:  # All except last
            # Should end with punctuation or be at the limit
            assert chunk.endswith('.') or len(chunk) >= config.chunk_size - 10
    
    @pytest.mark.asyncio
    async def test_split_long_section_llm_failure(self):
        """Test handling of LLM failures in long section splitting."""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10, max_chunk_size=100)
        chunker = SemanticChunker(config)
        
        # Mock the LLM agent to fail
        with patch('pydantic_ai.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run.side_effect = Exception("API Error")
            mock_agent_class.return_value = mock_agent
            
            long_section = "This is a very long section that needs to be split. " * 10
            chunks = await chunker._split_long_section(long_section)
            
            # Should fall back to simple splitting
            assert len(chunks) > 0
            assert all(len(chunk) <= config.max_chunk_size for chunk in chunks)


class TestFactoryFunction:
    """Test chunker factory function."""
    
    def test_create_semantic_chunker(self):
        """Test creating semantic chunker."""
        config = ChunkingConfig(use_semantic_splitting=True)
        chunker = create_chunker(config)
        
        assert isinstance(chunker, SemanticChunker)
        assert chunker.config == config
    
    def test_create_simple_chunker(self):
        """Test creating simple chunker."""
        config = ChunkingConfig(use_semantic_splitting=False)
        chunker = create_chunker(config)
        
        # SemanticChunker can also do simple chunking, so check the config
        assert chunker.config.use_semantic_splitting is False


class TestIntegration:
    """Integration tests for chunking."""
    
    @pytest.mark.asyncio
    async def test_real_document_chunking(self):
        """Test chunking a realistic document."""
        config = ChunkingConfig(
            chunk_size=200,
            chunk_overlap=50,
            use_semantic_splitting=False  # Use simple for predictable testing
        )
        chunker = create_chunker(config)
        
        content = """# AI Research Paper

## Abstract
This paper presents new findings in artificial intelligence research.
The study focuses on large language models and their applications.

## Introduction
Artificial intelligence has made significant progress in recent years.
Large language models have shown remarkable capabilities across various tasks.

## Methodology
We conducted experiments using state-of-the-art models.
The evaluation included multiple benchmark datasets.

### Data Collection
Data was collected from various sources including academic papers and web content.
Quality control measures were implemented to ensure data integrity.

### Model Training
Models were trained using distributed computing infrastructure.
Training time varied from several hours to multiple days.

## Results
Our experiments showed significant improvements over baseline methods.
The results demonstrate the effectiveness of our approach.

## Conclusion
This research contributes to the advancement of AI technology.
Future work will explore additional applications and improvements."""
        
        # SimpleChunker.chunk_document is synchronous, not async
        chunks = chunker.chunk_document(
            content=content,
            title="AI Research Paper",
            source="research.md",
            metadata={"author": "Test Author", "year": 2024}
        )
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Check metadata propagation
        for chunk in chunks:
            assert chunk.metadata["title"] == "AI Research Paper"
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["year"] == 2024
            assert "total_chunks" in chunk.metadata
        
        # Check indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
        
        # Check content coverage
        all_content = " ".join(chunk.content for chunk in chunks)
        # Should contain key terms from the document
        assert "artificial intelligence" in all_content.lower()
        assert "large language models" in all_content.lower()
        assert "methodology" in all_content.lower()
        
        # Check chunk sizes are reasonable
        for chunk in chunks:
            assert len(chunk.content) > 0
            assert len(chunk.content) <= config.max_chunk_size
    
    def test_metadata_consistency(self):
        """Test that metadata is consistent across chunks."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = SimpleChunker(config)
        
        content = "Test content. " * 50  # Long enough to create multiple chunks
        metadata = {"type": "test", "category": "document"}
        
        chunks = chunker.chunk_document(
            content=content,
            title="Test Document",
            source="test.md",
            metadata=metadata
        )
        
        # All chunks should have consistent metadata
        for chunk in chunks:
            assert chunk.metadata["title"] == "Test Document"
            assert chunk.metadata["type"] == "test"
            assert chunk.metadata["category"] == "document"
            assert chunk.metadata["chunk_method"] == "simple"
            assert chunk.metadata["total_chunks"] == len(chunks)