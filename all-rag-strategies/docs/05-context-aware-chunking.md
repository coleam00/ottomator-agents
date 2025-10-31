# Context-Aware Chunking

## Resource
**Semantic Chunking for RAG | Medium**
https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

## What It Is
Context-Aware Chunking (also called semantic chunking) intelligently determines chunk boundaries based on semantic similarity rather than fixed sizes. It generates embeddings for sentences, compares their similarity, and groups semantically related content together. This ensures chunks contain coherent topics, improving embedding quality and retrieval accuracy.

## Simple Example
```python
# Split document into sentences
sentences = document.split_sentences()

# Generate embeddings for each sentence
embeddings = [embed_model.encode(s) for s in sentences]

# Calculate similarity between consecutive sentences
similarities = [cosine_similarity(embeddings[i], embeddings[i+1])
                for i in range(len(embeddings)-1)]

# Group sentences where similarity > threshold
chunks = []
current_chunk = [sentences[0]]
for i, sim in enumerate(similarities):
    if sim > 0.8:  # High similarity = same topic
        current_chunk.append(sentences[i+1])
    else:  # Low similarity = new topic
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i+1]]
```

## Pros
Creates semantically coherent chunks that improve embedding quality. Preserves topic continuity and contextual meaning within chunks.

## Cons
Computationally expensive due to embedding every sentence. Slower indexing process compared to simple fixed-size chunking.

## When to Use It
Use when document topics are diverse and intermingled. Ideal for complex documents where topic boundaries are important for retrieval.

## When NOT to Use It
Avoid when processing speed is critical or documents are already well-structured. Skip for homogeneous documents with consistent topics throughout.
