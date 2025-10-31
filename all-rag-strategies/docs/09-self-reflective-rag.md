# Self-Reflective RAG

## Resource
**Self-Reflective RAG with LangGraph | LangChain**
https://blog.langchain.com/agentic-rag-with-langgraph/

## What It Is
Self-Reflective RAG (including Self-RAG and Corrective RAG/CRAG) adds self-assessment and iterative refinement to retrieval. The system evaluates whether retrieved documents are relevant, grades response quality, and refines queries or retrieves additional information when necessary. It creates a feedback loop where the system critiques its own outputs and adapts until producing a satisfactory answer.

## Simple Example
```python
# Initial retrieval
query = "What is quantum computing?"
docs = vector_search(query)

# Self-reflection: Grade document relevance
grades = []
for doc in docs:
    grade = llm.evaluate(f"Is this document relevant to '{query}'? {doc}")
    grades.append(grade)

# If relevance is low, refine and retry
if avg(grades) < 0.7:
    refined_query = llm.refine(query, docs)
    docs = vector_search(refined_query)

# Generate answer
answer = llm.generate(query, docs)

# Self-reflection: Verify answer quality
if not llm.verify_answer(answer, docs):
    # Retrieve more context or refine further
    additional_docs = web_search(query)
    answer = llm.generate(query, docs + additional_docs)
```

## Pros
Improves answer quality through self-correction and validation. Adapts dynamically to poor retrieval results by refining approach.

## Cons
Significantly higher latency due to multiple LLM calls and iterations. Increased cost and complexity compared to single-pass RAG.

## When to Use It
Use when answer accuracy is critical and errors are costly. Ideal for complex queries where initial retrieval may be insufficient.

## When NOT to Use It
Avoid when real-time responses are required. Skip for simple queries or when operating under strict latency/cost budgets.
