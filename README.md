# Enterprise RAG Knowledge System

Production Retrieval-Augmented Generation (RAG) system demonstrating modular AI architecture.

Pipeline:

documents → chunk → embed → vector search → retrieve → prompt → answer

---

## Run

pip install -r requirements.txt

uvicorn api.server:app --reload

---

## Example query

http://localhost:8000/query?q=remote work policy
