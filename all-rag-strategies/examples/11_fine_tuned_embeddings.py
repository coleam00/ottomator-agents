"""Fine-tuned Embeddings - Custom embedding model for domain-specific retrieval"""
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

agent = Agent('openai:gpt-4o', system_prompt='You are a RAG assistant with fine-tuned embeddings.')

conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

# Load fine-tuned embedding model (trained on domain data)
embedding_model = SentenceTransformer('./fine_tuned_model')  # Custom trained model

def prepare_training_data():
    """Create domain-specific query-document pairs"""
    training_pairs = [
        ("What is EBITDA?", "financial_doc_about_ebitda.txt"),
        ("Explain capital expenditure", "capex_explanation.txt"),
        # ... thousands more domain-specific pairs
    ]
    return training_pairs

def fine_tune_model():
    """Fine-tune on domain data (one-time process)"""
    base_model = SentenceTransformer('all-MiniLM-L6-v2')
    training_data = prepare_training_data()
    # Train with MultipleNegativesRankingLoss
    fine_tuned_model = base_model.fit(training_data, epochs=3)
    fine_tuned_model.save('./fine_tuned_model')

def get_embedding(text: str):
    """Use fine-tuned model for embeddings"""
    return embedding_model.encode(text)

def ingest_document(text: str):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = get_embedding(chunk)  # Uses fine-tuned model
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk, embedding))
    conn.commit()

@agent.tool
def search_knowledge_base(query: str) -> str:
    with conn.cursor() as cur:
        query_embedding = get_embedding(query)  # Uses fine-tuned model
        cur.execute('SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 3',
                   (query_embedding,))
        return "\n".join([row[0] for row in cur.fetchall()])

result = agent.run_sync("What is working capital?")
print(result.data)
