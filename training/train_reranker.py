"""Fit the reranker weights on the corpus relevance judgments.

Candidates come from the retriever, are labelled against the qrels, and a
logistic regression learns how to combine the reranking features. The features
are deliberately independent of the retrieval score, so the reranker can
genuinely disagree with the retriever rather than re-expressing its ranking.

The script reports nDCG@10 before and after reranking on a held-out query split.
**If reranking does not improve the held-out figure, that is reported and the
artifact is still written** -- the honest outcome is a measured null result, not
a tuned-until-positive one. `rag/reranker.py` no-ops without an artifact.

    python training/train_reranker.py
    python training/train_reranker.py --verify
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.metrics import evaluate_run  # noqa: E402
from rag.reranker import FEATURE_NAMES, features  # noqa: E402
from rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from rag.retrievers.dense import DEFAULT_MODEL  # noqa: E402
from rag.beir_data import embedding_cache_path  # noqa: E402

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "reranker.json"
CANDIDATE_DEPTH = 20
RANDOM_STATE = 42


def load_corpus():
    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))
    return corpus, queries


def build_examples(corpus, queries, retriever):
    documents = [f"{d['title']}. {d['text']}" for d in corpus]
    retriever.index(documents)

    per_query = {}
    for query in queries:
        relevant = set(query["relevant"])
        hits = retriever.search(query["text"], CANDIDATE_DEPTH)
        rows = []
        for rank, (index, score) in enumerate(hits):
            rows.append({
                "features": features(
                    query["text"], corpus[index]["text"], rank, corpus[index]["title"]
                ),
                "label": 1 if corpus[index]["id"] in relevant else 0,
                "doc_id": corpus[index]["id"],
                "rank": rank,
            })
        per_query[query["id"]] = rows
    return per_query


def ndcg_for(per_query, query_ids, qrels, order_key):
    run = {}
    for query_id in query_ids:
        rows = sorted(per_query[query_id], key=order_key, reverse=True)
        run[query_id] = [row["doc_id"] for row in rows[:10]]
    subset = {q: qrels[q] for q in query_ids}
    return evaluate_run(run, subset)["ndcg@10"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--retriever", default="hybrid")
    args = parser.parse_args()

    corpus, queries = load_corpus()
    qrels = {q["id"]: {d: 1 for d in q["relevant"]} for q in queries}

    cache = embedding_cache_path("synthetic", DEFAULT_MODEL)
    if args.retriever == "hybrid":
        retriever = HybridRetriever(
            BM25Retriever(), DenseRetriever(model_name=DEFAULT_MODEL, cache_path=cache)
        )
    else:
        retriever = BM25Retriever()

    per_query = build_examples(corpus, queries, retriever)

    query_ids = [q["id"] for q in queries]
    train_ids, test_ids = train_test_split(
        query_ids, test_size=0.3, random_state=RANDOM_STATE
    )

    x_train = np.array([r["features"] for q in train_ids for r in per_query[q]])
    y_train = np.array([r["label"] for q in train_ids for r in per_query[q]])

    if len(set(y_train)) < 2:
        print("FAIL: training candidates contain only one class")
        return 1

    model = LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    weights = {name: float(w) for name, w in zip(FEATURE_NAMES, model.coef_[0])}
    intercept = float(model.intercept_[0])

    def fitted_score(row):
        return intercept + sum(
            weights[name] * value for name, value in zip(FEATURE_NAMES, row["features"])
        )

    baseline = ndcg_for(per_query, test_ids, qrels, lambda r: -r["rank"])
    reranked = ndcg_for(per_query, test_ids, qrels, fitted_score)
    delta = round(reranked - baseline, 4)

    payload = {
        "note": "Fitted by training/train_reranker.py on corpus relevance judgments.",
        "retriever": retriever.name,
        "candidate_depth": CANDIDATE_DEPTH,
        "n_train_queries": len(train_ids),
        "n_test_queries": len(test_ids),
        "intercept": intercept,
        "weights": weights,
        "held_out_ndcg10_before": baseline,
        "held_out_ndcg10_after": reranked,
        "held_out_delta": delta,
        "improves": bool(delta > 0),
    }

    if args.verify:
        if not ARTIFACT_PATH.exists():
            print("FAIL: reranker artifact missing; run without --verify first")
            return 1
        committed = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        if abs(committed["held_out_delta"] - delta) > 0.05:
            print(f"FAIL: reranker delta drifted {committed['held_out_delta']} -> {delta}")
            return 1
        print(f"OK: reranker delta {delta} matches committed {committed['held_out_delta']}")
        return 0

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"retriever          : {retriever.name}")
    print(f"train/test queries : {len(train_ids)}/{len(test_ids)}")
    print(f"held-out nDCG@10   : {baseline} -> {reranked}  (delta {delta:+.4f})")
    for name, value in sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True):
        print(f"  {name:20s} {value:+.4f}")
    if delta <= 0:
        print(
            "\nNOTE: reranking did NOT improve held-out nDCG@10. Reported as-is.\n"
            "      See the README -- the claim is worded to match this result."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
