Flagship RAG architecture

Hybrid Retrieval:

vector similarity
+
keyword scoring

Pipeline:

documents
→ semantic chunking
→ embeddings
→ hybrid retrieval
→ reranking
→ reasoning layer
→ structured answer

Evaluation:

precision@k

Extensible components:

embedding providers
retrievers
rerankers
LLM generators
