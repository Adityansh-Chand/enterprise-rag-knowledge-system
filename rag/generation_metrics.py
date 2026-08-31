"""Metrics for the answer, not the ranking.

Retrieval metrics stop at "was the right document returned". Everything after
that -- whether the answer used it, whether it said the thing that mattered,
whether it declined a question the corpus cannot answer -- is unmeasured by
nDCG, and is where a retrieval system actually fails a user.

Every metric here is computed against the corpus and the judgments, never by
asking a model to grade itself. That constraint is the reason this harness costs
nothing to run and produces the same numbers on every machine. It is also the
reason the metrics are lexical, which is a real limitation and is stated as one
in the model card rather than glossed.

The five:

  groundedness        share of answer terms traceable to retrieved context.
                      Near 1.0 for an extractive generator by construction --
                      it copies sentences. Reported anyway, precisely because a
                      near-perfect score here certifies almost nothing on its
                      own. It earns its keep on the LLM path.

  attribution         share of the answer's sentences that came from a chunk
                      judged relevant. This one can fail: retrieval returns a
                      wrong document, extraction happily quotes it, and
                      groundedness still reads 1.0.

  fact_coverage       share of the gold content units the answer conveys. The
                      question "did it say the thing that mattered", which is
                      not the same question as "is it grounded".

  abstention          on questions the corpus cannot answer, did the system
                      decline. Retrieval always returns its top k, so something
                      plausible is always available to quote.

  context_utilisation share of retrieved chunks that contributed a sentence.
                      Low means paying to retrieve and encode text that does
                      not reach the answer -- a cost signal, not a quality one.
"""
from rag.retrievers.bm25 import tokenize

# The BM25 tokenizer keeps trailing punctuation ("receipt." stays one token) and
# keeps underscores inside identifiers. That is right for retrieval -- it is
# what preserves `retry.max_attempts` -- but it makes "receipt" and "receipt."
# different terms, which would silently break every comparison here. Normalised
# at this boundary rather than in the tokenizer, because changing the tokenizer
# would change every committed retrieval number for a cosmetic reason.
_EDGE = "./-_"


def terms(text):
    """Content terms of a string, with identifier-preserving normalisation."""
    return {t.strip(_EDGE) for t in tokenize(text)} - {""}


def groundedness(answer, chunks):
    """Share of answer terms that appear somewhere in the retrieved context."""
    answer_terms = terms(answer)
    if not answer_terms:
        return 0.0
    return len(answer_terms & terms(" ".join(chunks))) / len(answer_terms)


def fact_coverage(answer, facts):
    """Share of gold content units the answer conveys.

    A unit counts only when every one of its terms is present -- partial credit
    for half a remediation step would make the metric agreeable and useless.
    """
    if not facts:
        return 0.0
    answer_terms = terms(answer)
    covered = sum(1 for fact in facts if all(t in answer_terms for t in fact))
    return covered / len(facts)


def attribution(sentence_sources, relevant_ids):
    """Share of answer sentences drawn from a chunk judged relevant.

    `sentence_sources` is [doc_id, ...], one per sentence in the answer, in the
    order they appear. Returns 0.0 for an empty answer: nothing attributable.
    """
    if not sentence_sources:
        return 0.0
    relevant = set(relevant_ids)
    return sum(1 for doc_id in sentence_sources if doc_id in relevant) / len(
        sentence_sources
    )


def context_utilisation(sentence_sources, retrieved_ids):
    """Share of retrieved chunks that contributed at least one answer sentence."""
    if not retrieved_ids:
        return 0.0
    return len(set(sentence_sources) & set(retrieved_ids)) / len(set(retrieved_ids))


def separation_auc(positive, negative):
    """Probability a random positive scores above a random negative.

    Used to compare abstention signals without first choosing a threshold, so
    the comparison is about the signal rather than about an operating point.
    0.5 is a coin flip. Ties are credited half, which matters here because
    BM25 scores whole groups of queries at exactly zero and a tie-blind
    implementation would silently flatter whichever signal produced them.
    """
    if not positive or not negative:
        return 0.0

    pairs = sorted(
        [(value, 1) for value in positive] + [(value, 0) for value in negative],
        key=lambda item: item[0],
    )
    rank_sum, index = 0.0, 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(1 for k in range(index, end) if pairs[k][1] == 1)
        index = end

    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarise(rows, keys):
    """Mean of each key across rows, rounded. Empty input gives zeros, not a crash."""
    if not rows:
        return {key: 0.0 for key in keys}
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 4) for key in keys
    }
