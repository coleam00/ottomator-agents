"""Query Expansion RAG - Generate multiple query variations for better retrieval"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

# Initialize agent
agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with query expansion.')

# Database connection
conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

# Ingest documents (simplified)
def ingest_document(text: str):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Simple chunking
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = get_embedding(chunk)  # Assume embedding function exists
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk, embedding))
    conn.commit()

@agent.tool
def expand_query(query: str) -> list[str]:
    """Expand single query into multiple variations"""
    expansion_prompt = f"Generate 3 different variations of this query: '{query}'"
    # LLM generates variations
    variations = ["original query", "rephrased query 1", "rephrased query 2"]
    return variations

@agent.tool
def search_knowledge_base(queries: list[str]) -> str:
    """Search vector DB with multiple query variations"""
    all_results = []
    with conn.cursor() as cur:
        for query in queries:
            query_embedding = get_embedding(query)
            cur.execute(
                'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                (query_embedding,)
            )
            all_results.extend([row[0] for row in cur.fetchall()])
    return "\n".join(set(all_results))  # Deduplicate

# Run agent
result = agent.run_sync("What is machine learning?")
print(result.data)
