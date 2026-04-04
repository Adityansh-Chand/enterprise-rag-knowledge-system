import json
from rag.pipeline import RAGPipeline


pipeline = RAGPipeline()

pipeline.ingest_document(

open("datasets/knowledge_base/hr_policy.txt").read()

)


data = json.load(

open("datasets/eval_queries.json")

)


correct = 0

for row in data:

    result = pipeline.query(row["query"])

    if row["expected"].lower() in result["answer"].lower():

        correct += 1


print("accuracy", correct/len(data))
