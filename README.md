
# Enterprise RAG Knowledge System

Production Retrieval-Augmented Generation pipeline designed using modular AI architecture patterns.

## Architecture

```mermaid
flowchart LR
Documents --> Chunker
Chunker --> Embedder
Embedder --> VectorDB
VectorDB --> Retriever
Retriever --> Reranker
Reranker --> Generator
Generator --> Evaluator
```

## Pipeline
documents → chunk → embed → retrieve → rerank → generate → evaluate

### Highlights
semantic chunking, 
reranking abstraction, 
confidence scoring and 
evaluation scaffold.

## License
MIT
