"""Re-ranking RAG - Two-stage retrieval with cross-encoder"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with re-ranking.')

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
def search_with_reranking(query: str) -> str:
    """Two-stage: fast retrieval + accurate reranking"""
    # Stage 1: Fast vector search (retrieve 20 candidates)
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute(
            'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 20',
            (query_embedding,)
        )
        candidates = [row[0] for row in cur.fetchall()]

    # Stage 2: Re-rank with cross-encoder
    scored_results = []
    for doc in candidates:
        score = cross_encoder_score(query, doc)  # Assume cross-encoder function
        scored_results.append((doc, score))

    # Return top 5 after re-ranking
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return "\n".join([doc for doc, _ in scored_results[:5]])

# Run agent
result = agent.run_sync("Explain neural networks")
print(result.data)
