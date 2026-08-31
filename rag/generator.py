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
from rag import abstention, generation_metrics
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
    """Rank sentences across retrieved chunks by query term coverage.

    Returns [(sentence, chunk_rank, coverage), ...]. `coverage` is the raw share
    of query terms the sentence matches, without the rank prior -- the prior is
    a tie-break for ordering and would make the number mean something else if it
    leaked into the abstention decision.
    """
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
            scored.append(
                (coverage + 0.05 / (chunk_rank + 1), sentence, chunk_rank, coverage)
            )

    scored.sort(key=lambda item: item[0], reverse=True)

    # Drop repeats. The corpus contains the same runbook per service and region
    # deliberately, so the top chunks are often near-duplicates carrying an
    # identical sentence -- and without this the answer spent all three of its
    # slots restating one sentence three times. Found by the fact-coverage
    # metric, which cannot rise while two thirds of the answer is a copy.
    selected, seen = [], set()
    for _, sentence, rank, coverage in scored:
        key = sentence.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append((sentence, rank, coverage))
        if len(selected) == limit:
            break
    return selected


def groundedness(answer, chunks):
    """Share of the answer's content terms that appear in the retrieved context.

    1.0 means every term is traceable to something retrieved. This is a
    verifiable lexical property, not a model's opinion of its own output -- and
    it is the number that makes "grounded" a checkable claim rather than a label.

    Worth knowing what it does not say: an extractive answer scores near 1.0 by
    construction, because it copies sentences. See `rag/generation_metrics.py`.
    """
    return round(generation_metrics.groundedness(answer, chunks), 4)


def generate_answer(query, chunks, use_llm=True, scores=None, abstain=True,
                    retriever=None, signal=None, threshold=None):
    """Return (answer_text, mode). Falls back to extraction on any LLM failure.

    `signal` and `threshold` pin the abstention operating point explicitly,
    which is what the bench does so a reported number is tied to a stated
    threshold rather than to whatever artifact happened to be on disk.
    """
    if not chunks:
        return NO_ANSWER, "no_context"

    # Checked before the LLM call, not after. A question the corpus cannot
    # answer is where a generator is most likely to invent something, and it is
    # also the call worth not paying for.
    if abstain and abstention.should_abstain(
        query, chunks, scores, signal=signal, threshold=threshold, retriever=retriever
    ):
        return NO_ANSWER, "abstained"

    selected = select_sentences(query, chunks)

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

    if not selected:
        return NO_ANSWER, "extractive"
    return " ".join(sentence for sentence, _, _ in selected), "extractive"


def build_response(query, results, retriever=None):
    """Assemble the API response.

    `results` is [(score, text, title, doc_id), ...] in final rank order.
    `retriever` selects which fitted abstention threshold applies, since the
    signal is a raw retrieval score and its scale differs per retriever.
    """
    if not results:
        return {
            "answer": NO_ANSWER,
            "mode": "no_context",
            "groundedness": 0.0,
            "retrieval_score": 0.0,
            "answer_sources": [],
            "sources": [],
        }

    chunks = [row[1] for row in results]
    answer, mode = generate_answer(
        query, chunks, scores=[row[0] for row in results], retriever=retriever
    )

    # Which chunk each answer sentence came from, in order. Only meaningful when
    # the answer is extracted: an LLM rewrites, so sentence-to-chunk identity no
    # longer holds and claiming otherwise would be a fabricated citation.
    if mode == "extractive":
        doc_ids = [row[3] if len(row) > 3 else None for row in results]
        answer_sources = [
            doc_ids[rank]
            for _, rank, _ in select_sentences(query, chunks)
            if rank < len(doc_ids)
        ]
    else:
        answer_sources = []

    return {
        "answer": answer,
        "mode": mode,
        "groundedness": groundedness(answer, chunks),
        "answer_sources": answer_sources,
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
