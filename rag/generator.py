"""Answer construction from retrieved context, with a groundedness measure.

The previous implementation returned
`f"Based on the knowledge base: {first_line_of_context}"` -- it echoed the top
chunk's first line regardless of what was asked, and reported the retrieval
score under the name "confidence".

This selects the sentences that actually respond to the query, cites the chunks
they came from, and reports how much of the answer is traceable to retrieved
text. An optional LLM path is available through the provider-agnostic seam; the
extractive path is the default and is what every reported metric uses.
"""
import re

from llm import client as llm
from rag.retrievers.bm25 import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

MAX_ANSWER_SENTENCES = 3
NO_ANSWER = "No relevant information was found in the knowledge base."

SYSTEM_PROMPT = (
    "Answer strictly from the provided context. If the context does not contain "
    "the answer, say so. Do not use outside knowledge. Be concise."
)


def _sentences(text):
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def select_sentences(query, chunks, limit=MAX_ANSWER_SENTENCES):
    """Rank sentences across retrieved chunks by query term coverage."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    scored = []
    for chunk_rank, chunk in enumerate(chunks):
        for sentence in _sentences(chunk):
            terms = set(tokenize(sentence))
            if not terms:
                continue
            coverage = len(query_terms & terms) / len(query_terms)
            # Small prior toward higher-ranked chunks, so an equally-matching
            # sentence from a better chunk wins.
            scored.append((coverage + 0.05 / (chunk_rank + 1), sentence, chunk_rank))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(s, r) for _, s, r in scored[:limit]]


def groundedness(answer, chunks):
    """Share of the answer's content terms that appear in the retrieved context.

    1.0 means every term is traceable to something retrieved. This is a
    verifiable lexical property, not a model's opinion of its own output -- and
    it is the number that makes "grounded" a checkable claim rather than a label.
    """
    answer_terms = set(tokenize(answer))
    if not answer_terms:
        return 0.0
    context_terms = set(tokenize(" ".join(chunks)))
    return round(len(answer_terms & context_terms) / len(answer_terms), 4)


def generate_answer(query, chunks, use_llm=True):
    """Return (answer_text, mode). Falls back to extraction on any LLM failure."""
    if not chunks:
        return NO_ANSWER, "no_context"

    if use_llm and llm.is_enabled():
        context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
        try:
            answer = llm.complete(
                SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {query}"
            ).strip()
            if answer:
                return answer, f"llm:{llm.provider()}"
        except llm.LLMError:
            pass  # deterministic path below

    selected = select_sentences(query, chunks)
    if not selected:
        return NO_ANSWER, "extractive"
    return " ".join(sentence for sentence, _ in selected), "extractive"


def build_response(query, results):
    """Assemble the API response.

    `results` is [(score, text, title, doc_id), ...] in final rank order.
    """
    if not results:
        return {
            "answer": NO_ANSWER,
            "mode": "no_context",
            "groundedness": 0.0,
            "retrieval_score": 0.0,
            "sources": [],
        }

    chunks = [row[1] for row in results]
    answer, mode = generate_answer(query, chunks)

    return {
        "answer": answer,
        "mode": mode,
        "groundedness": groundedness(answer, chunks),
        # Deliberately NOT called "confidence": it is the top retrieval score,
        # which says how well the query matched, not how likely the answer is right.
        "retrieval_score": round(float(results[0][0]), 4),
        "sources": [
            {
                "doc_id": row[3] if len(row) > 3 else None,
                "title": row[2] if len(row) > 2 else None,
                "text": row[1],
                "score": round(float(row[0]), 4),
            }
            for row in results
        ],
    }
