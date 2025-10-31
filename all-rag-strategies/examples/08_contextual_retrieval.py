"""Contextual Retrieval - Add document context to chunks (Anthropic)"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('anthropic:claude-3-5-sonnet', system_prompt='You are a RAG assistant.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def add_context_to_chunk(document: str, chunk: str) -> str:
    """Use LLM to generate chunk-specific context"""
    prompt = f"""Document: {document[:500]}...

Chunk: {chunk}

Provide brief context explaining what this chunk is about in relation to the document."""

    context = llm_generate(prompt)  # Returns: "This chunk is from..."
    return f"{context} {chunk}"

def ingest_document(text: str):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    with conn.cursor() as cur:
        for chunk in chunks:
            # Add contextual prefix to chunk
            contextualized_chunk = add_context_to_chunk(text, chunk)

            # Embed contextualized version
            embedding = get_embedding(contextualized_chunk)
            cur.execute(
                'INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                (contextualized_chunk, embedding)
            )
    conn.commit()

@agent.tool
def search_knowledge_base(query: str) -> str:
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                   (query_embedding,))
        return "\n".join([row[0] for row in cur.fetchall()])

result = agent.run_sync("What were Q2 earnings?")
print(result.data)
