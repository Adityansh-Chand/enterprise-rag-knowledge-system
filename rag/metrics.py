"""Standard ranking metrics, computed against relevance judgments.

The previous `rag/evaluator.py` defined `precision_at_k(results, expected)` which
ignored `k` entirely, joined every retrieved document into one string, and
substring-matched. That is not precision@k and could not distinguish a system
that ranked the answer first from one that ranked it last. These are the real
definitions.

`qrels` maps query_id -> {doc_id: relevance}, where relevance > 0 means relevant.
`run` maps query_id -> [doc_id, ...] in rank order.
"""
import math


def recall_at_k(run, qrels, k: int) -> float:
    """Fraction of relevant documents that appear in the top k."""
    totals = []
    for query_id, relevant in qrels.items():
        if not relevant:
            continue
        retrieved = set(run.get(query_id, [])[:k])
        totals.append(len(retrieved & set(relevant)) / len(relevant))
    return sum(totals) / len(totals) if totals else 0.0


def mrr_at_k(run, qrels, k: int) -> float:
    """Mean reciprocal rank of the first relevant document."""
    totals = []
    for query_id, relevant in qrels.items():
        if not relevant:
            continue
        score = 0.0
        for rank, doc_id in enumerate(run.get(query_id, [])[:k], start=1):
            if relevant.get(doc_id, 0) > 0:
                score = 1.0 / rank
                break
        totals.append(score)
    return sum(totals) / len(totals) if totals else 0.0


def ndcg_at_k(run, qrels, k: int) -> float:
    """Normalised discounted cumulative gain -- graded, rank-sensitive."""
    totals = []
    for query_id, relevant in qrels.items():
        if not relevant:
            continue

        gains = [
            relevant.get(doc_id, 0) for doc_id in run.get(query_id, [])[:k]
        ]
        dcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1) if g > 0)

        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1) if g > 0)

        totals.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(totals) / len(totals) if totals else 0.0


def evaluate_run(run, qrels, ks=(1, 5, 10)) -> dict:
    """All metrics at several cut-offs, for one retriever on one dataset."""
    results = {"n_queries": len([q for q, r in qrels.items() if r])}
    for k in ks:
        results[f"recall@{k}"] = round(recall_at_k(run, qrels, k), 4)
        results[f"ndcg@{k}"] = round(ndcg_at_k(run, qrels, k), 4)
    results["mrr@10"] = round(mrr_at_k(run, qrels, 10), 4)
    return results
