"""Select the lexical/semantic fusion weight on a dev split of queries.

Unweighted reciprocal rank fusion scored *below* its own dense component on both
corpora — 0.7913 vs 0.8577 on the synthetic set, 0.3423 vs 0.3727 on
BEIR/NFCorpus. Giving a much weaker retriever equal say drags the fusion down.
The fix is to let the balance be set from data.

The point of this script is that the weight is chosen **honestly**:

- Queries are split into dev and report halves, deterministically by query id.
- The weight sweep only ever sees the **dev** half.
- The chosen weight is then applied once to the **report** half, which was never
  used for selection.

Tuning the weight on the same queries used to report would manufacture an
improvement — it would be the same circularity this repository exists to remove.
The gap between the dev-best and report scores is printed so overfitting to dev
is visible rather than hidden.

    python training/tune_fusion.py                  # synthetic corpus
    python training/tune_fusion.py --beir nfcorpus  # BEIR subset
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.beir_data import embedding_cache_path, load_beir  # noqa: E402
from rag.metrics import evaluate_run  # noqa: E402
from rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from rag.retrievers.dense import DEFAULT_MODEL  # noqa: E402

ARTIFACT_PATH = ROOT / "models" / "artifacts" / "fusion_weights.json"
DEPTH = 10

# Sweep the semantic share; lexical weight is 1 - semantic. 0.5 is the
# unweighted baseline, 1.0 is dense-only.
SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_dataset(name):
    if name:
        corpus, queries, qrels = load_beir(name)
        documents = [
            (f"{d['title']}. {d['text']}" if d["title"] else d["text"]) for d in corpus
        ]
        doc_ids = [d["id"] for d in corpus]
        cache = embedding_cache_path(name, DEFAULT_MODEL)
        return documents, doc_ids, queries, qrels, cache

    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    raw = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))
    documents = [f"{d['title']}. {d['text']}" for d in corpus]
    doc_ids = [d["id"] for d in corpus]
    queries = [{"id": q["id"], "text": q["text"]} for q in raw]
    qrels = {q["id"]: {d: 1 for d in q["relevant"]} for q in raw}
    cache = embedding_cache_path("synthetic", DEFAULT_MODEL)
    return documents, doc_ids, queries, qrels, cache


def split_queries(queries):
    """Deterministic dev/report halves by sorted query id -- no shuffling seed."""
    ordered = sorted(queries, key=lambda q: q["id"])
    dev = [q for index, q in enumerate(ordered) if index % 2 == 0]
    report = [q for index, q in enumerate(ordered) if index % 2 == 1]
    return dev, report


def score(retriever, doc_ids, queries, qrels):
    run = {
        q["id"]: [doc_ids[i] for i, _ in retriever.search(q["text"], DEPTH)]
        for q in queries
    }
    subset = {q["id"]: qrels[q["id"]] for q in queries if q["id"] in qrels}
    return evaluate_run(run, subset)["ndcg@10"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--save", action="store_true",
                        help="write the chosen weight to models/artifacts/")
    args = parser.parse_args()

    documents, doc_ids, queries, qrels, cache = load_dataset(args.beir)
    dev, report = split_queries(queries)
    label = args.beir or "synthetic"

    print(f"dataset      : {label}  ({len(documents)} docs)")
    print(f"query split  : {len(dev)} dev / {len(report)} report "
          f"(report queries are never used for selection)")
    print()

    lexical = BM25Retriever()
    semantic = DenseRetriever(model_name=args.model, cache_path=cache)
    lexical.index(documents)
    semantic.index(documents)

    # Reference points, scored on the report half only.
    bm25_report = score(lexical, doc_ids, report, qrels)
    dense_report = score(semantic, doc_ids, report, qrels)

    print("sweeping semantic weight on the DEV half:")
    dev_scores = {}
    for semantic_weight in SWEEP:
        fusion = HybridRetriever(
            lexical, semantic,
            lexical_weight=round(1.0 - semantic_weight, 3),
            semantic_weight=semantic_weight,
        )
        dev_scores[semantic_weight] = score(fusion, doc_ids, dev, qrels)
        marker = "  <- unweighted RRF" if semantic_weight == 0.5 else ""
        print(f"  semantic={semantic_weight:.1f}  dev nDCG@10 {dev_scores[semantic_weight]:.4f}{marker}")

    best_weight = max(dev_scores, key=dev_scores.get)
    tuned = HybridRetriever(
        lexical, semantic,
        lexical_weight=round(1.0 - best_weight, 3), semantic_weight=best_weight,
    )
    unweighted = HybridRetriever(lexical, semantic)

    tuned_report = score(tuned, doc_ids, report, qrels)
    unweighted_report = score(unweighted, doc_ids, report, qrels)

    print()
    print(f"chosen on dev: semantic weight {best_weight:.1f} "
          f"(dev nDCG@10 {dev_scores[best_weight]:.4f})")
    print()
    print("--- REPORT half (never used for selection) ---")
    print(f"  bm25 alone             {bm25_report:.4f}")
    print(f"  dense alone            {dense_report:.4f}")
    print(f"  hybrid, unweighted     {unweighted_report:.4f}")
    print(f"  hybrid, tuned weight   {tuned_report:.4f}")
    print()

    delta_vs_unweighted = tuned_report - unweighted_report
    delta_vs_dense = tuned_report - dense_report
    print(f"tuned vs unweighted : {delta_vs_unweighted:+.4f}")
    print(f"tuned vs dense alone: {delta_vs_dense:+.4f}")
    print(f"dev -> report drop  : {dev_scores[best_weight] - tuned_report:+.4f} "
          f"(large negative means the weight overfit dev)")
    print()
    if delta_vs_dense <= 0:
        print("Weighting did NOT lift fusion above dense alone on the report half.")
        print("Reported as-is; the README claim is worded to match.")
    else:
        print("Weighting lifted fusion above dense alone on held-out queries.")

    if args.save:
        payload = {
            "note": (
                "Semantic fusion weight selected on a dev half of queries and "
                "applied once to a disjoint report half. Never tuned on the "
                "queries used to report."
            ),
            "dataset": label,
            "semantic_weight": best_weight,
            "lexical_weight": round(1.0 - best_weight, 3),
            "dev_sweep": {str(k): round(v, 4) for k, v in dev_scores.items()},
            "report_half": {
                "bm25": round(bm25_report, 4),
                "dense": round(dense_report, 4),
                "hybrid_unweighted": round(unweighted_report, 4),
                "hybrid_tuned": round(tuned_report, 4),
            },
        }
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {ARTIFACT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
