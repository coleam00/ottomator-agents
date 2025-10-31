"""Hierarchical RAG - Parent-child chunks with metadata"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with hierarchical retrieval.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def ingest_document(text: str):
    # Create parent chunks (large sections)
    parent_chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]

    with conn.cursor() as cur:
        for parent_id, parent in enumerate(parent_chunks):
            # Store parent (not embedded)
            cur.execute('INSERT INTO parent_chunks (id, content) VALUES (%s, %s)',
                       (parent_id, parent))

            # Create child chunks from parent
            child_chunks = [parent[j:j+500] for j in range(0, len(parent), 500)]
            for child in child_chunks:
                embedding = get_embedding(child)
                # Store child with parent_id reference
                cur.execute(
                    'INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)',
                    (child, embedding, parent_id)
                )
    conn.commit()

@agent.tool
def search_knowledge_base(query: str) -> str:
    """Search children, return parents for context"""
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)

        # Find matching child chunks
        cur.execute(
            'SELECT parent_id FROM child_chunks ORDER BY embedding <=> %s LIMIT 3',
            (query_embedding,)
        )
        parent_ids = [row[0] for row in cur.fetchall()]

        # Retrieve full parent chunks
        cur.execute(
            'SELECT content FROM parent_chunks WHERE id = ANY(%s)',
            (parent_ids,)
        )
        return "\n".join([row[0] for row in cur.fetchall()])

result = agent.run_sync("What is backpropagation?")
print(result.data)
