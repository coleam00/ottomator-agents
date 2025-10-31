"""Multi-Query RAG - Parallel searches with multiple reformulations"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with multi-query retrieval.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def ingest_document(text: str):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = get_embedding(chunk)
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk, embedding))
    conn.commit()

@agent.tool
def multi_query_search(original_query: str) -> str:
    """Generate multiple query perspectives and search in parallel"""
    # Generate query variations (LLM generates these)
    queries = [
        original_query,
        "rephrased version 1",
        "rephrased version 2",
        "related query angle"
    ]

    all_results = set()
    with conn.cursor() as cur:
        for query in queries:
            query_embedding = get_embedding(query)
            cur.execute(
                'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 5',
                (query_embedding,)
            )
            all_results.update([row[0] for row in cur.fetchall()])

    # Return unique union of all results
    return "\n".join(all_results)

# Run agent
result = agent.run_sync("How do I deploy ML models?")
print(result.data)
