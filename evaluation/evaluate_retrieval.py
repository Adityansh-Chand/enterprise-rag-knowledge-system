from rag.pipeline import RAGPipeline


p = RAGPipeline()

p.ingest_document(

    "Employees receive 20 days vacation annually."

)

print(

    p.query("How many vacation days?")
)
