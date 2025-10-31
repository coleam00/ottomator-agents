"""Late Chunking - Embed full document first, then chunk token embeddings"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with late chunking.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def late_chunk(text: str, chunk_size=512) -> list[tuple[str, list]]:
    """Process full document through transformer BEFORE chunking"""
    # Step 1: Embed entire document (up to 8192 tokens)
    full_doc_token_embeddings = transformer_embed(text)  # Returns token-level embeddings

    # Step 2: Define chunk boundaries
    tokens = text.split()  # Simplified tokenization
    chunk_boundaries = range(0, len(tokens), chunk_size)

    # Step 3: Pool token embeddings for each chunk
    chunks_with_embeddings = []
    for i, start in enumerate(chunk_boundaries):
        end = start + chunk_size
        chunk_text = ' '.join(tokens[start:end])

        # Mean pool the token embeddings (preserves full doc context)
        chunk_embedding = mean_pool(full_doc_token_embeddings[start:end])
        chunks_with_embeddings.append((chunk_text, chunk_embedding))

    return chunks_with_embeddings

def ingest_document(text: str):
    chunks = late_chunk(text)
    with conn.cursor() as cur:
        for chunk_text, embedding in chunks:
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk_text, embedding))
    conn.commit()

@agent.tool
def search_knowledge_base(query: str) -> str:
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                   (query_embedding,))
        return "\n".join([row[0] for row in cur.fetchall()])

result = agent.run_sync("Explain transformers")
print(result.data)
