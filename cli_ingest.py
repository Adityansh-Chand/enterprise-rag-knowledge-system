import sys
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

for file in sys.argv[1:]:

    with open(file) as f:

        pipeline.ingest_document(f.read())

print("Documents ingested")
