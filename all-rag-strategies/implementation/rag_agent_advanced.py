"""
Advanced RAG CLI Agent with Multiple Strategies
===============================================
Implements multiple RAG strategies:
- Query Expansion
- Re-ranking
- Agentic RAG (semantic search + full file retrieval)
- Multi-Query RAG
- Self-Reflective RAG
- Context-aware chunking (via Docling HybridChunker - already in ingestion)
"""

import asyncio
import asyncpg
import json
import logging
import os
import sys
from typing import Any, List, Dict
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Global database pool
db_pool = None

# Initialize cross-encoder for re-ranking
reranker = None


async def initialize_db():
    """Initialize database connection pool."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


def initialize_reranker():
    """Initialize cross-encoder model for re-ranking."""
    global reranker
    if reranker is None:
        logger.info("Loading cross-encoder model for re-ranking...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder loaded")


# ======================
# STRATEGY 1: QUERY EXPANSION
# ======================

async def expand_query_variations(ctx: RunContext[None], query: str) -> List[str]:
    """
    Generate multiple variations of a query for better retrieval.

    Args:
        query: Original search query

    Returns:
        List of query variations including the original
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    expansion_prompt = f"""Generate 3 different variations of this search query.
Each variation should capture a different perspective or phrasing while maintaining the same intent.

Original query: {query}

Return only the 3 variations, one per line, without numbers or bullets."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": expansion_prompt}],
            temperature=0.7
        )

        variations_text = response.choices[0].message.content.strip()
        variations = [v.strip() for v in variations_text.split('\n') if v.strip()]

        # Return original + variations
        return [query] + variations[:3]

    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query]  # Fallback to original query


# ======================
# STRATEGY 2 & 3: MULTI-QUERY RAG (parallel search with variations)
# ======================

async def search_with_multi_query(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Search using multiple query variations in parallel (Multi-Query RAG).

    This combines query expansion with parallel execution for better recall.

    Args:
        query: The search query
        limit: Results per query variation

    Returns:
        Formatted deduplicated search results
    """
    try:
        if not db_pool:
            await initialize_db()

        # Generate query variations
        queries = await expand_query_variations(ctx, query)
        logger.info(f"Multi-query search with {len(queries)} variations")

        # Generate embeddings for all queries
        from ingestion.embedder import create_embedder
        embedder = create_embedder()

        # Execute searches in parallel
        all_results = []
        search_tasks = []

        async with db_pool.acquire() as conn:
            for q in queries:
                query_embedding = await embedder.embed_query(q)
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

                task = conn.fetch(
                    """
                    SELECT * FROM match_chunks($1::vector, $2)
                    """,
                    embedding_str,
                    limit
                )
                search_tasks.append(task)

            # Execute all searches concurrently
            results_lists = await asyncio.gather(*search_tasks)

            # Collect all results
            for results in results_lists:
                all_results.extend(results)

        if not all_results:
            return "No relevant information found."

        # Deduplicate by chunk ID and keep highest similarity
        seen = {}
        for row in all_results:
            chunk_id = row['chunk_id']
            if chunk_id not in seen or row['similarity'] > seen[chunk_id]['similarity']:
                seen[chunk_id] = row

        unique_results = sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)[:limit]

        # Format results
        response_parts = []
        for i, row in enumerate(unique_results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Multi-query search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ======================
# STRATEGY 3: RE-RANKING
# ======================

async def search_with_reranking(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Two-stage retrieval: Fast vector search + precise cross-encoder re-ranking.

    Args:
        query: The search query
        limit: Final number of results to return after re-ranking

    Returns:
        Formatted re-ranked search results
    """
    try:
        if not db_pool:
            await initialize_db()

        initialize_reranker()

        # Stage 1: Fast vector retrieval (retrieve more candidates)
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Retrieve 20 candidates for re-ranking
        candidate_limit = min(limit * 4, 20)

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                candidate_limit
            )

        if not results:
            return "No relevant information found."

        # Stage 2: Re-rank with cross-encoder
        logger.info(f"Re-ranking {len(results)} candidates")

        pairs = [[query, row['content']] for row in results]
        scores = reranker.predict(pairs)

        # Combine results with new scores
        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        # Format results
        response_parts = []
        for i, (row, score) in enumerate(reranked, 1):
            response_parts.append(
                f"[Source: {row['document_title']} | Relevance: {score:.2f}]\n{row['content']}\n"
            )

        return f"Found {len(response_parts)} highly relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Re-ranking search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ======================
# STRATEGY 4: AGENTIC RAG (Semantic Search + Full File Retrieval)
# ======================

async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Standard semantic search over chunks.

    Args:
        query: The search query
        limit: Maximum number of results

    Returns:
        Formatted search results
    """
    try:
        if not db_pool:
            await initialize_db()

        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        if not results:
            return "No relevant information found in the knowledge base for your query."

        response_parts = []
        for i, row in enumerate(results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


async def retrieve_full_document(ctx: RunContext[None], document_title: str) -> str:
    """
    Retrieve the full content of a specific document by title.

    Use this when chunks don't provide enough context or when you need
    to see the complete document.

    Args:
        document_title: The title of the document to retrieve

    Returns:
        Full document content
    """
    try:
        if not db_pool:
            await initialize_db()

        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT title, content, source
                FROM documents
                WHERE title ILIKE $1
                LIMIT 1
                """,
                f"%{document_title}%"
            )

        if not result:
            # Try to list available documents
            async with db_pool.acquire() as conn:
                docs = await conn.fetch(
                    """
                    SELECT title FROM documents
                    ORDER BY created_at DESC
                    LIMIT 10
                    """
                )

            doc_list = "\n- ".join([doc['title'] for doc in docs])
            return f"Document '{document_title}' not found. Available documents:\n- {doc_list}"

        return f"**Document: {result['title']}**\n\nSource: {result['source']}\n\n{result['content']}"

    except Exception as e:
        logger.error(f"Full document retrieval failed: {e}", exc_info=True)
        return f"Error retrieving document: {str(e)}"


# ======================
# STRATEGY 5: SELF-REFLECTIVE RAG
# ======================

async def search_with_self_reflection(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Self-reflective search: evaluate results and refine if needed.

    This implements a simple self-reflection loop:
    1. Perform initial search
    2. Grade relevance of results
    3. If results are poor, refine query and search again

    Args:
        query: The search query
        limit: Number of results to return

    Returns:
        Formatted search results with reflection metadata
    """
    try:
        if not db_pool:
            await initialize_db()

        from openai import AsyncOpenAI
        from ingestion.embedder import create_embedder

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embedder = create_embedder()

        # Initial search
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        if not results:
            return "No relevant information found."

        # Self-reflection: Grade relevance
        grade_prompt = f"""Query: {query}

Retrieved Documents:
{chr(10).join([f"{i+1}. {r['content'][:200]}..." for i, r in enumerate(results)])}

Grade the overall relevance of these documents to the query on a scale of 1-5:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Relevant
5 = Highly relevant

Respond with only a single number (1-5) and a brief reason."""

        try:
            grade_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": grade_prompt}],
                temperature=0
            )

            grade_text = grade_response.choices[0].message.content.strip()
            grade_score = int(grade_text.split()[0])

        except Exception as e:
            logger.warning(f"Grading failed, proceeding with results: {e}")
            grade_score = 3  # Assume moderate relevance

        # If relevance is low, refine query
        if grade_score < 3:
            logger.info(f"Low relevance score ({grade_score}), refining query")

            refine_prompt = f"""The query "{query}" returned low-relevance results.
Suggest an improved, more specific query that might find better results.
Respond with only the improved query, nothing else."""

            try:
                refine_response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": refine_prompt}],
                    temperature=0.7
                )

                refined_query = refine_response.choices[0].message.content.strip()
                logger.info(f"Refined query: {refined_query}")

                # Search again with refined query
                refined_embedding = await embedder.embed_query(refined_query)
                refined_embedding_str = '[' + ','.join(map(str, refined_embedding)) + ']'

                async with db_pool.acquire() as conn:
                    results = await conn.fetch(
                        """
                        SELECT * FROM match_chunks($1::vector, $2)
                        """,
                        refined_embedding_str,
                        limit
                    )

                reflection_note = f"\n[Reflection: Refined query from '{query}' to '{refined_query}']\n"

            except Exception as e:
                logger.warning(f"Query refinement failed: {e}")
                reflection_note = "\n[Reflection: Initial results had low relevance]\n"
        else:
            reflection_note = f"\n[Reflection: Results deemed relevant (score: {grade_score}/5)]\n"

        # Format final results
        response_parts = []
        for i, row in enumerate(results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return reflection_note + f"Found {len(response_parts)} results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Self-reflective search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ======================
# CREATE AGENT WITH ALL STRATEGIES
# ======================

agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an advanced knowledge assistant with multiple retrieval strategies at your disposal.

AVAILABLE TOOLS:
1. search_knowledge_base - Standard semantic search over document chunks
2. retrieve_full_document - Get complete document when chunks aren't enough
3. search_with_multi_query - Use multiple query variations for better recall
4. search_with_reranking - Use two-stage retrieval with re-ranking for precision
5. search_with_self_reflection - Evaluate and refine search results automatically

STRATEGY SELECTION GUIDE:
- Use search_knowledge_base for most queries (fast, reliable)
- Use retrieve_full_document when you need full context or found relevant chunks but need more
- Use search_with_multi_query when query is ambiguous or could be interpreted multiple ways
- Use search_with_reranking when precision is critical (legal, medical, financial queries)
- Use search_with_self_reflection for complex research questions

You can use multiple tools in sequence if needed. Be concise but thorough.""",
    tools=[
        search_knowledge_base,
        retrieve_full_document,
        search_with_multi_query,
        search_with_reranking,
        search_with_self_reflection
    ]
)


async def run_cli():
    """Run the agent in an interactive CLI with streaming."""

    await initialize_db()

    print("=" * 60)
    print("Advanced RAG Knowledge Assistant")
    print("=" * 60)
    print("Multiple retrieval strategies available!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            print("Assistant: ", end="", flush=True)

            try:
                async with agent.run_stream(
                    user_input,
                    message_history=message_history
                ) as result:
                    async for text in result.stream_text(delta=True):
                        print(text, end="", flush=True)

                    print()
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Agent error: {e}", exc_info=True)

            print()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await close_db()


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")
