import re


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_document(text, max_sentences=3, overlap=1):
    """Chunk text by sentence with light overlap for retrieval continuity."""
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    step = max(1, max_sentences - overlap)
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start:start + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_sentences >= len(sentences):
            break

    return chunks
