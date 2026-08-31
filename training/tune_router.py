"""Does per-query routing beat picking one retriever globally?

Weighted fusion chose "pure dense" on both corpora. A single global weight cannot
exploit BM25 being better on identifier-shaped queries specifically, so this tests
whether choosing per query does better than choosing once.

Method, mirroring `training/tune_fusion.py`:

- queries are split into a **dev** half and a **report** half by sorted id
- the routing rule is fixed in `rag/retrievers/router.py` before any measurement
- dev is used only to observe how often the rule fires and on what
- every headline number comes from the report half

The rule sees the query text only. The corpus labels queries by type, and routing
on those labels would score beautifully and prove nothing -- it is the circular
evaluation this repository exists to have removed.

    python training/tune_router.py
    python training/tune_router.py --verify
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.harness import DEFAULT_MODEL, DEPTH, embedding_cache_path  # noqa: E402
from rag.metrics import evaluate_run  # noqa: E402
from rag.retrievers.bm25 import BM25Retriever  # noqa: E402
from rag.retrievers.dense import DenseRetriever  # noqa: E402
from rag.retrievers.hybrid import HybridRetriever  # noqa: E402
from rag.retrievers.router import RouterRetriever, looks_lexical  # noqa: E402
from training.tune_fusion import split_queries  # noqa: E402

RESULTS_PATH = ROOT / "models" / "artifacts" / "router_comparison.json"


def score(retriever, doc_ids, queries, qrels):
    run = {
        query["id"]: [doc_ids[i] for i, _ in retriever.search(query["text"], DEPTH)]
        for query in queries
    }
    return evaluate_run(run, {q["id"]: qrels[q["id"]] for q in queries})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))
    documents = [f"{d['title']}. {d['text']}" for d in corpus]
    doc_ids = [d["id"] for d in corpus]
    qrels = {q["id"]: {doc_id: 1 for doc_id in q["relevant"]} for q in queries}

    dev, report = split_queries(queries)

    bm25 = BM25Retriever()
    dense = DenseRetriever(
        model_name=args.model,
        cache_path=embedding_cache_path("synthetic", args.model),
    )
    router = RouterRetriever(BM25Retriever(), DenseRetriever(
        model_name=args.model,
        cache_path=embedding_cache_path("synthetic", args.model),
    ))
    hybrid = HybridRetriever(BM25Retriever(), DenseRetriever(
        model_name=args.model,
        cache_path=embedding_cache_path("synthetic", args.model),
    ))

    for retriever in (bm25, dense, router, hybrid):
        retriever.index(documents)

    # Dev half: what the rule does, not how well it scores.
    dev_routing = Counter(
        "lexical" if looks_lexical(q["text"]) else "semantic" for q in dev
    )
    dev_by_type = Counter(
        (q["type"], "lexical" if looks_lexical(q["text"]) else "semantic") for q in dev
    )

    arms = {}
    for name, retriever in (("bm25", bm25), ("dense", dense),
                            ("router", router), ("hybrid", hybrid)):
        arms[name] = score(retriever, doc_ids, report, qrels)

    # Per query type on the report half, which is where routing should show up if
    # it works at all.
    types = sorted({q["type"] for q in report})
    by_type = {}
    for query_type in types:
        subset = [q for q in report if q["type"] == query_type]
        by_type[query_type] = {
            name: score(retriever, doc_ids, subset, qrels)["ndcg@10"]
            for name, retriever in (("bm25", bm25), ("dense", dense),
                                    ("router", router))
        }

    results = {
        "split": "dev/report halves by sorted query id; report half only below",
        "n_dev": len(dev),
        "n_report": len(report),
        "rule": "query containing an identifier-shaped token -> lexical, else semantic",
        "dev_routing": dict(dev_routing),
        "dev_routing_by_type": {f"{t}:{d}": n for (t, d), n in sorted(dev_by_type.items())},
        "report": arms,
        "report_by_query_type": by_type,
        "router_vs_dense_ndcg": round(
            arms["router"]["ndcg@10"] - arms["dense"]["ndcg@10"], 4
        ),
    }

    if args.verify:
        if not RESULTS_PATH.exists():
            print("FAIL: router_comparison.json missing; run without --verify first")
            return 1
        committed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        old = committed["report"]["router"]["ndcg@10"]
        new = arms["router"]["ndcg@10"]
        if abs(old - new) > 0.02:
            print(f"FAIL: router nDCG@10 drifted {old} -> {new}")
            return 1
        print(f"OK: router nDCG@10 {new} matches committed {old}")
        return 0

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"dev routing: {dict(dev_routing)}")
    print(f"  by type  : {results['dev_routing_by_type']}")
    print(f"\n=== report half (n={len(report)}) ===")
    header = f"{'retriever':<12} {'nDCG@10':>8} {'Recall@10':>10} {'MRR@10':>8}"
    print(header)
    print("-" * len(header))
    for name in ("bm25", "dense", "router", "hybrid"):
        row = arms[name]
        print(f"{name:<12} {row['ndcg@10']:>8.4f} {row['recall@10']:>10.4f} "
              f"{row['mrr@10']:>8.4f}")

    print(f"\n--- nDCG@10 by query type (report half) ---")
    print(f"{'query type':<22}{'bm25':>10}{'dense':>10}{'router':>10}")
    for query_type in types:
        scores = by_type[query_type]
        print(f"{query_type:<22}{scores['bm25']:>10.4f}{scores['dense']:>10.4f}"
              f"{scores['router']:>10.4f}")

    print(f"\nrouter vs dense: {results['router_vs_dense_ndcg']:+.4f} nDCG@10")
    print(f"written: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
