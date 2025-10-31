"""Context-Aware Chunking - Semantic boundaries using embedding similarity"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with semantic chunking.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def semantic_chunk(text: str, similarity_threshold=0.8) -> list[str]:
    """Chunk based on semantic similarity, not fixed size"""
    sentences = text.split('. ')  # Simple sentence split
    sentence_embeddings = [get_embedding(s) for s in sentences]

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(sentences) - 1):
        similarity = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i+1])

        if similarity > similarity_threshold:  # Same topic
            current_chunk.append(sentences[i+1])
        else:  # Topic boundary detected
            chunks.append('. '.join(current_chunk))
            current_chunk = [sentences[i+1]]

    chunks.append('. '.join(current_chunk))
    return chunks

def ingest_document(text: str):
    chunks = semantic_chunk(text)  # Semantic chunking
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = get_embedding(chunk)
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk, embedding))
    conn.commit()

@agent.tool
def search_knowledge_base(query: str) -> str:
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                   (query_embedding,))
        return "\n".join([row[0] for row in cur.fetchall()])

result = agent.run_sync("What is deep learning?")
print(result.data)
