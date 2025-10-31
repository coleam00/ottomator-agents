"""Agentic RAG - Agent dynamically chooses tools (vector, SQL, web)"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='You are an agentic RAG assistant with multiple tools.')

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
def vector_search(query: str) -> str:
    """Search unstructured knowledge base"""
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                   (query_embedding,))
        return "\n".join([row[0] for row in cur.fetchall()])

@agent.tool
def sql_query(question: str) -> str:
    """Query structured database for specific data"""
    # Agent can write SQL for structured queries
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM sales WHERE quarter='Q2'")  # Example
        return str(cur.fetchall())

@agent.tool
def web_search(query: str) -> str:
    """Search web for external information"""
    return f"Web results for: {query}"  # Simplified

# Agent autonomously picks which tool(s) to use
result = agent.run_sync("What were ACME Corp's Q2 sales?")
print(result.data)
