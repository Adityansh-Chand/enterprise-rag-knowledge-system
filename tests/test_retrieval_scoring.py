from rag.pipeline import RAGPipeline


def test_confidence_score():

    p = RAGPipeline()

    p.ingest_document(
        "Remote work allowed with approval."
    )

    r = p.query("remote policy")

    assert r["confidence"] >= 0
