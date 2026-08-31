"""Does reranking help here at all, and was the fitted reranker just too weak?

`rag/reranker.py` -- features fitted by logistic regression on the relevance
judgments -- measured **+0.0000 nDCG@10**. That is published rather than buried,
but it leaves an unanswered question: is reranking worthless on this corpus, or
was that model simply not strong enough to show a difference? Those have opposite
implications and only a stronger reranker separates them.

So three arms, same retriever, same candidates, same qrels:

    1. no reranking            the retriever's own order
    2. fitted pairwise         rag/reranker.py
    3. cross-encoder           ms-marco-MiniLM-L-6-v2, Apache 2.0

Run on both tracks. The synthetic corpus is ours; NFCorpus has human relevance
judgments and is the one that counts.

    python evaluation/rerank_bench.py                  # synthetic
    python evaluation/rerank_bench.py --beir nfcorpus  # real judgments
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.harness import (  # noqa: E402
    DEFAULT_MODEL,
    DEPTH,
    embedding_cache_path,
    make_retrievers,
)
from rag import cross_encoder  # noqa: E402
from rag import reranker as fitted_reranker  # noqa: E402
from rag.metrics import evaluate_run  # noqa: E402

RESULTS_PATH = ROOT / "models" / "artifacts" / "rerank_comparison.json"


def candidates_for(retriever, documents, doc_ids, query_text, depth):
    """(score, text, doc_id) -- the convention both rerankers already use.

    Score first, text second, extra fields carried through: the same shape
    rag/pipeline.py passes, so neither reranker needs an adapter here.
    """
    return [
        (score, documents[index], doc_ids[index])
        for index, score in retriever.search(query_text, depth)
    ]


def run_arm(name, rerank_fn, retriever, documents, doc_ids, queries, qrels):
    run, elapsed = {}, 0.0
    for query in queries:
        candidates = candidates_for(retriever, documents, doc_ids, query["text"], DEPTH)
        start = time.time()
        ordered = rerank_fn(query["text"], candidates) if rerank_fn else candidates
        elapsed += time.time() - start
        run[query["id"]] = [doc_id for _, _, doc_id in ordered]

    metrics = evaluate_run(run, qrels)
    return {
        "arm": name,
        **metrics,
        "rerank_ms_per_query": round(elapsed / max(len(queries), 1) * 1000, 1),
    }


def compare(documents, doc_ids, queries, qrels, retriever, label):
    retriever.index(documents)

    arms = [run_arm("no reranking", None, retriever, documents, doc_ids, queries, qrels)]

    if fitted_reranker.is_fitted():
        arms.append(run_arm("fitted pairwise", fitted_reranker.rerank,
                            retriever, documents, doc_ids, queries, qrels))
    else:
        print("fitted reranker has no artifact; run training/train_reranker.py")

    if cross_encoder.available():
        arms.append(run_arm(f"cross-encoder ({cross_encoder.model_name().split('/')[-1]})",
                            cross_encoder.rerank,
                            retriever, documents, doc_ids, queries, qrels))
    else:
        # Reported rather than silently skipped: an arm that did not run must not
        # look like an arm that ran and tied.
        print("cross-encoder unavailable (sentence-transformers or weights missing); "
              "that arm is SKIPPED, not zero")

    baseline = arms[0]["ndcg@10"]
    for arm in arms:
        arm["delta_ndcg@10"] = round(arm["ndcg@10"] - baseline, 4)

    header = (f"{'arm':<34} {'nDCG@10':>8} {'delta':>8} {'Recall@10':>10} "
              f"{'MRR@10':>8} {'ms/query':>9}")
    print(f"\n=== {label} ===")
    print(header)
    print("-" * len(header))
    for arm in arms:
        print(f"{arm['arm']:<34} {arm['ndcg@10']:>8.4f} {arm['delta_ndcg@10']:>+8.4f} "
              f"{arm['recall@10']:>10.4f} {arm['mrr@10']:>8.4f} "
              f"{arm['rerank_ms_per_query']:>9.1f}")
    return arms


def synthetic(model_name):
    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))
    documents = [f"{d['title']}. {d['text']}" for d in corpus]
    doc_ids = [d["id"] for d in corpus]
    qrels = {q["id"]: {doc_id: 1 for doc_id in q["relevant"]} for q in queries}

    retriever = make_retrievers(
        ["dense"], embedding_cache_path("synthetic", model_name), model_name
    )[0]
    return compare(documents, doc_ids, queries, qrels, retriever,
                   "Synthetic corpus (ours; NOT a benchmark)")


def beir(dataset, model_name):
    from rag.beir_data import load_beir

    corpus, queries, qrels = load_beir(dataset)
    # Same field assembly as evaluation/harness.py, so the two are measuring the
    # same documents rather than two different renderings of them.
    documents = [
        (f"{d['title']}. {d['text']}" if d["title"] else d["text"]) for d in corpus
    ]
    doc_ids = [d["id"] for d in corpus]

    retriever = make_retrievers(
        ["dense"], embedding_cache_path(dataset, model_name), model_name
    )[0]
    return compare(documents, doc_ids, queries, qrels, retriever,
                   f"BEIR/{dataset} (human relevance judgments)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beir", help="run against a BEIR subset, e.g. nfcorpus")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    results = {"retriever": "dense", "depth": DEPTH,
               "cross_encoder_model": cross_encoder.model_name()}
    if args.beir:
        results[args.beir] = beir(args.beir, args.model)
    else:
        results["synthetic"] = synthetic(args.model)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    existing.update(results)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
