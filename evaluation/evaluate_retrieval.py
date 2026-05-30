import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

pipeline.ingest_document(
    "Employees receive 20 days annual leave."
)

result = pipeline.query("vacation policy")

print(result["answer"])
print("confidence", result["confidence"])
