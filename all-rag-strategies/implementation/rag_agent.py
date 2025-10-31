"""
RAG CLI Agent with PostgreSQL/PGVector
=======================================
Text-based CLI agent that searches through knowledge base using semantic similarity
"""

import asyncio
import asyncpg
import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Global database pool
db_pool = None


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


async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source citations
    """
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        # Generate embedding for query
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)

        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Search using match_chunks function
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        # Format results for response
        if not results:
            return "No relevant information found in the knowledge base for your query."

        # Build response with sources
        response_parts = []
        for i, row in enumerate(results, 1):
            similarity = row['similarity']
            content = row['content']
            doc_title = row['document_title']
            doc_source = row['document_source']

            response_parts.append(
                f"[Source: {doc_title}]\n{content}\n"
            )

        if not response_parts:
            return "Found some results but they may not be directly relevant to your query. Please try rephrasing your question."

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"I encountered an error searching the knowledge base: {str(e)}"


# Create the PydanticAI agent with the RAG tool
agent = Agent(
    'openai:gpt-4o.1-mini',
    system_prompt="""You are an intelligent knowledge assistant with access to an organization's documentation and information.
Your role is to help users find accurate information from the knowledge base.
You have a professional yet friendly demeanor.

IMPORTANT: Always search the knowledge base before answering questions about specific information.
If information isn't in the knowledge base, clearly state that and offer general guidance.
Be concise but thorough in your responses.
Ask clarifying questions if the user's query is ambiguous.
When you find relevant information, synthesize it clearly and cite the source documents.""",
    tools=[search_knowledge_base]
)


async def run_cli():
    """Run the agent in an interactive CLI with streaming."""

    # Initialize database
    await initialize_db()

    print("=" * 60)
    print("RAG Knowledge Assistant")
    print("=" * 60)
    print("Ask me anything about the knowledge base!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            print("Assistant: ", end="", flush=True)

            try:
                # Stream the response using run_stream
                async with agent.run_stream(
                    user_input,
                    message_history=message_history
                ) as result:
                    # Stream text as it comes in (delta=True for only new tokens)
                    async for text in result.stream_text(delta=True):
                        # Print only the new token
                        print(text, end="", flush=True)

                    print()  # New line after streaming completes

                    # Update message history for context
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Agent error: {e}", exc_info=True)

            print()  # Extra line for readability

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await close_db()


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    # Run the CLI
    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")