"""
Contextual Retrieval Enhancement (Anthropic Method)
===================================================
Adds document-level context to each chunk before embedding to improve retrieval accuracy.

This implements Anthropic's Contextual Retrieval approach where each chunk is enriched
with explanatory context about what it discusses in relation to the whole document.

Benefits:
- Reduces retrieval failures by 35-49%
- Chunks become more self-contained and meaningful
- Better for complex documents where context matters
"""

import asyncio
import logging
from typing import List, Optional
import os

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ContextualEnricher:
    """Adds contextual information to document chunks."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize contextual enricher.

        Args:
            model: LLM model to use for context generation
        """
        self.model = model
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def enrich_chunk(
        self,
        chunk_content: str,
        document_content: str,
        document_title: str,
        document_source: str
    ) -> str:
        """
        Add contextual prefix to a chunk.

        Args:
            chunk_content: The chunk text to enrich
            document_content: Full document content (or large excerpt)
            document_title: Document title
            document_source: Document source/filename

        Returns:
            Enriched chunk with contextual prefix
        """
        # Limit document content to avoid token limits
        document_excerpt = document_content[:4000] if len(document_content) > 4000 else document_content

        prompt = f"""<document>
Title: {document_title}
Source: {document_source}

{document_excerpt}
</document>

<chunk>
{chunk_content}
</chunk>

Provide a brief, 1-2 sentence context explaining what this chunk discusses in relation to the overall document.
The context should help someone understand this chunk without seeing the full document.

Format your response as:
"This chunk from [document title] discusses [brief explanation]."

Be concise and specific. Do not include any preamble or explanation, just the context sentence(s)."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150
            )

            context = response.choices[0].message.content.strip()

            # Combine context with chunk
            enriched_chunk = f"{context}\n\n{chunk_content}"

            return enriched_chunk

        except Exception as e:
            logger.error(f"Failed to enrich chunk: {e}")
            # Fallback to simple context
            fallback_context = f"This chunk is from the document '{document_title}' ({document_source})."
            return f"{fallback_context}\n\n{chunk_content}"

    async def enrich_chunks_batch(
        self,
        chunks: List[str],
        document_content: str,
        document_title: str,
        document_source: str,
        max_concurrent: int = 5
    ) -> List[str]:
        """
        Enrich multiple chunks with contextual information (with concurrency control).

        Args:
            chunks: List of chunk texts to enrich
            document_content: Full document content
            document_title: Document title
            document_source: Document source
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of enriched chunks
        """
        if not chunks:
            return []

        logger.info(f"Enriching {len(chunks)} chunks with contextual information...")

        # Create tasks with semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def enrich_with_semaphore(chunk: str):
            async with semaphore:
                return await self.enrich_chunk(
                    chunk,
                    document_content,
                    document_title,
                    document_source
                )

        # Execute all enrichments concurrently (but rate-limited)
        enriched_chunks = await asyncio.gather(
            *[enrich_with_semaphore(chunk) for chunk in chunks],
            return_exceptions=True
        )

        # Handle any exceptions
        final_chunks = []
        for i, result in enumerate(enriched_chunks):
            if isinstance(result, Exception):
                logger.error(f"Failed to enrich chunk {i}: {result}")
                # Use fallback
                fallback = f"This chunk is from '{document_title}'.\n\n{chunks[i]}"
                final_chunks.append(fallback)
            else:
                final_chunks.append(result)

        logger.info(f"Successfully enriched {len(final_chunks)} chunks")
        return final_chunks


# Factory function
def create_contextual_enricher(model: str = "gpt-4o-mini") -> ContextualEnricher:
    """
    Create a contextual enricher instance.

    Args:
        model: LLM model to use

    Returns:
        ContextualEnricher instance
    """
    return ContextualEnricher(model=model)


# Example usage
async def main():
    """Example usage of contextual enrichment."""
    enricher = create_contextual_enricher()

    # Sample document
    doc_content = """Machine Learning Best Practices

Introduction to ML
Machine learning requires careful attention to data quality, model selection, and evaluation metrics.

Data Preparation
Clean data is essential. Remove duplicates, handle missing values, and normalize features.

Model Training
Start with simple models before trying complex ones. Use cross-validation to avoid overfitting."""

    chunks = [
        "Clean data is essential. Remove duplicates, handle missing values, and normalize features.",
        "Start with simple models before trying complex ones. Use cross-validation to avoid overfitting."
    ]

    enriched = await enricher.enrich_chunks_batch(
        chunks,
        doc_content,
        "Machine Learning Best Practices",
        "ml_guide.md"
    )

    for i, chunk in enumerate(enriched):
        print(f"\n=== Chunk {i+1} ===")
        print(chunk)
        print()


if __name__ == "__main__":
    asyncio.run(main())
