"""Self-Reflective RAG - Iteratively refine with self-assessment"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='You are a self-reflective RAG assistant.')

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
def search_and_grade(query: str) -> dict:
    """Retrieve and self-grade relevance"""
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 5',
                   (query_embedding,))
        docs = [row[0] for row in cur.fetchall()]

    # Self-reflection: grade document relevance
    relevant_docs = []
    for doc in docs:
        grade = llm_grade_relevance(query, doc)  # Returns 0-1
        if grade > 0.7:
            relevant_docs.append(doc)

    return {"docs": relevant_docs, "quality": len(relevant_docs) / len(docs)}

@agent.tool
def refine_query(original_query: str, docs: list) -> str:
    """Refine query if initial results are poor"""
    return llm_refine(original_query, docs)  # Returns improved query

@agent.tool
def answer_with_verification(query: str, context: str) -> str:
    """Generate and verify answer quality"""
    answer = llm_generate(query, context)
    is_supported = llm_verify(answer, context)  # Check if grounded
    return answer if is_supported else "Need more context"

# Agent runs iterative loop
result = agent.run_sync("What is quantum computing?")
print(result.data)
