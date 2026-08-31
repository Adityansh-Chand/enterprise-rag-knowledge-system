"""Service configuration.

`EMBEDDING_PROVIDER` used to live here with the single value "local", raising on
anything else -- a stub shaped like a plug-in point. It is replaced by RETRIEVER,
which selects among four genuinely different implementations.
"""
import os

# bm25 | lsa | dense | hybrid  -- see rag/retrievers/
# Default is bm25 so a fresh clone starts instantly and the smoke test needs no
# model download. The dense and hybrid retrievers score materially better (see
# the bench in the README) -- switch with RETRIEVER=router or RETRIEVER=dense,
# which download the
# embedding model on first use.
RETRIEVER = os.getenv("RETRIEVER", "bm25")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
TOP_K = int(os.getenv("TOP_K", "5"))

# Candidates fetched before reranking. Reranking cannot recover a document the
# retriever never returned, so this is deliberately wider than TOP_K.
RERANK_DEPTH = int(os.getenv("RERANK_DEPTH", "20"))
