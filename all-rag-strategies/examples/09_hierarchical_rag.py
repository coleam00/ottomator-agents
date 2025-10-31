"""Hierarchical RAG - Search small chunks, return big parents with metadata"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector
import json

agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with hierarchical retrieval.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

def ingest_document(text: str, doc_title: str):
    # Create parent chunks (large sections)
    parent_chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]

    with conn.cursor() as cur:
        for parent_id, parent in enumerate(parent_chunks):
            # Store parent with simple metadata
            metadata = json.dumps({"heading": f"{doc_title} - Section {parent_id}", "type": "detail"})
            cur.execute('INSERT INTO parent_chunks (id, content, metadata) VALUES (%s, %s, %s)',
                       (parent_id, parent, metadata))

            # Create child chunks from parent
            child_chunks = [parent[j:j+500] for j in range(0, len(parent), 500)]
            for child in child_chunks:
                embedding = get_embedding(child)
                cur.execute(
                    'INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)',
                    (child, embedding, parent_id)
                )
    conn.commit()

@agent.tool
def search_knowledge_base(query: str) -> str:
    """Search children, return parents with heading context"""
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)

        # Find matching children and join with parent metadata
        cur.execute(
            '''SELECT p.content, p.metadata
               FROM child_chunks c
               JOIN parent_chunks p ON c.parent_id = p.id
               ORDER BY c.embedding <=> %s LIMIT 3''',
            (query_embedding,)
        )

        # Return parents with heading context
        results = []
        for content, metadata_json in cur.fetchall():
            metadata = json.loads(metadata_json)
            results.append(f"[{metadata['heading']}]\n{content}")

        return "\n\n".join(results)

result = agent.run_sync("What is backpropagation?")
print(result.data)
