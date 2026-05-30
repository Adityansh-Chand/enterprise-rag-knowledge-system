from rag.pipeline import RAGPipeline


def test_pipeline():

    p = RAGPipeline()

    p.ingest_document(

        "Remote work allowed with approval."

    )

    r = p.query("Can I work remotely?")

    assert "Remote work allowed" in r["answer"]
    assert r["sources"]
