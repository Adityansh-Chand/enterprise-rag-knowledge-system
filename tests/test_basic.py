from rag.chunker import chunk_document
from rag.embedder import embed, tokenize


def test_chunker_preserves_sentence_text():
    chunks = chunk_document("Remote work is allowed. Security training is annual.")

    assert chunks == ["Remote work is allowed. Security training is annual."]


def test_local_embedding_has_stable_shape():
    vector = embed("remote work policy")

    assert vector.shape == (128,)
    assert vector.sum() > 0


def test_tokenizer_normalizes_common_hr_terms():
    assert tokenize("Can I work remotely during vacation?") == ["work", "remote", "leave"]
