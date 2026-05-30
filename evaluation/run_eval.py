import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

pipeline.ingest_document((ROOT / "datasets" / "knowledge_base" / "hr_policy.txt").read_text(encoding="utf-8"))


data = json.loads((ROOT / "datasets" / "eval_queries.json").read_text(encoding="utf-8"))


correct = 0

for row in data:

    result = pipeline.query(row["query"])

    if row["expected"].lower() in result["answer"].lower():

        correct += 1


print("accuracy", correct/len(data))
