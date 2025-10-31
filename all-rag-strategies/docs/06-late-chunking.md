# Late Chunking

## Resource
**Late Chunking in Long-Context Embedding Models | Jina AI**
https://jina.ai/news/late-chunking-in-long-context-embedding-models/

## What It Is
Late Chunking processes entire documents (or large sections) through the embedding model's transformer before splitting into chunks. Traditional "naive chunking" splits text first, losing long-distance context. Late Chunking embeds all tokens together, then applies chunking after the transformer but before pooling, preserving full contextual information in each chunk's embedding.

## Simple Example
```python
# Traditional chunking (loses context)
chunks = split_document(doc, chunk_size=512)
embeddings = [embed_model.encode(chunk) for chunk in chunks]

# Late Chunking (preserves context)
# 1. Process entire document through transformer
full_doc_embeddings = transformer_layer(doc)  # 8192 tokens max

# 2. Chunk the token embeddings (not the text)
chunk_boundaries = [0, 512, 1024, 1536, ...]
chunk_embeddings = []
for i in range(len(chunk_boundaries)-1):
    start, end = chunk_boundaries[i], chunk_boundaries[i+1]
    # Mean pool the token embeddings for this chunk
    chunk_emb = mean_pool(full_doc_embeddings[start:end])
    chunk_embeddings.append(chunk_emb)
```

## Pros
Maintains full document context in chunk embeddings, improving accuracy. Leverages long-context models (8K+ tokens) effectively.

## Cons
Requires long-context embedding models with high token limits. More complex implementation than standard chunking approaches.

## When to Use It
Use when document context is crucial for understanding chunks. Ideal for documents where meaning depends on long-distance relationships.

## When NOT to Use It
Avoid when using standard embedding models with small context windows. Skip if documents are already short and context is local.
