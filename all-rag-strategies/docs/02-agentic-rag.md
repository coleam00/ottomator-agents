# Agentic RAG

## Resource
**What is Agentic RAG? Building Agents with Qdrant**
https://qdrant.tech/articles/agentic-rag/

## What It Is
Agentic RAG empowers autonomous agents with multiple tools to explore knowledge dynamically. Unlike traditional RAG's single vector search, agents can write SQL queries for structured data, perform web searches, read entire files, or query multiple vector stores based on query complexity. The agent reasons about which tools to use and in what order, adapting its strategy to the task.

## Simple Example
```python
# Agent decides which tools to use
agent = RAGAgent(tools=[vector_search, sql_query, web_search, file_reader])

query = "What were Q2 sales for ACME Corp?"

# Agent reasoning:
# 1. Checks if structured data needed → use SQL tool
result = agent.sql_tool("SELECT revenue FROM quarterly_sales WHERE company='ACME' AND quarter='Q2'")

# If insufficient, agent tries another tool
if not result.complete:
    result += agent.vector_search("ACME Q2 financial performance")
```

## Pros
Highly flexible and adapts to query complexity. Can access heterogeneous data sources (SQL, files, web, vectors) intelligently.

## Cons
Increased complexity and unpredictability in behavior. Higher latency and cost due to multi-step reasoning and tool calls.

## When to Use It
Use for complex queries requiring multiple data sources or exploration strategies. Ideal when you have diverse knowledge types (structured and unstructured).

## When NOT to Use It
Avoid for simple lookups where traditional RAG suffices. Skip when you need predictable, fast responses or have limited tool infrastructure.
