"""Ingest text files into a pipeline and run one query against them.

    python cli_ingest.py notes.txt "what does the runbook say about retries"
"""
import sys

from rag.pipeline import RAGPipeline


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    *paths, question = sys.argv[1:]
    pipeline = RAGPipeline()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            chunks = pipeline.ingest_document(handle.read(), title=path, doc_id=path)
        print(f"ingested {path} -> {chunks} chunks")

    pipeline.build_index()
    response = pipeline.query(question)
    print(f"\nretriever   : {response['retriever']}")
    print(f"groundedness: {response['groundedness']}")
    print(f"answer      : {response['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
