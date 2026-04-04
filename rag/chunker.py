import re


def split_sentences(text):

    # split on sentence boundaries

    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [

        s.strip()

        for s in sentences

        if len(s.strip()) > 10

    ]


def chunk_document(text, max_sentences=3):

    sentences = split_sentences(text)

    chunks = []

    for i in range(0, len(sentences), max_sentences):

        chunk = " ".join(sentences[i:i+max_sentences])

        chunks.append(chunk)

    return chunks
