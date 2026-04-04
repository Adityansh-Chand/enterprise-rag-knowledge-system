from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

pipeline.ingest_document(
    "Employees receive 20 days annual leave."
)

result = pipeline.query("vacation policy")

print(result)
