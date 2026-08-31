"""One evaluation harness, every retriever, both data tracks.

Track A (BEIR) carries the headline numbers: real corpora with human relevance
judgments, comparable to published results. Track B (the synthetic
mixed-enterprise corpus) runs offline and breaks results down by query type,
which is where the interesting disagreement between methods shows up.

    python evaluation/harness.py                    # synthetic corpus (offline)
    python evaluation/harness.py --beir nfcorpus    # one BEIR subset
    python evaluation/harness.py --beir all         # every configured subset
    python evaluation/harness.py --retrievers bm25,lsa
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.beir_data import DATASETS, embedding_cache_path, load_beir  # noqa: E402
from rag.metrics import evaluate_run  # noqa: E402
from rag.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LSARetriever,
)
from rag.retrievers.dense import DEFAULT_MODEL  # noqa: E402

RESULTS_PATH = ROOT / "evaluation" / "results.json"
LATENCY_PATH = ROOT / "evaluation" / "latency.json"
DEPTH = 10

# Timings are printed and written to `latency.json`, which is not committed.
# A quality metric is a property of the method and is identical on any machine;
# a latency is a property of the machine, and committing it meant every run
# produced a diff that said nothing. The durable timing record lives with the
# cost model, where it is deliberately refreshed and stamped with the hardware
# it was measured on, rather than churning on every evaluation.
TIMING_KEYS = ("index_seconds", "query_ms")


def make_retrievers(names, cache_path=None, model_name=DEFAULT_MODEL):
    def dense():
        return DenseRetriever(model_name=model_name, cache_path=cache_path)

    available = {
        "bm25": lambda: BM25Retriever(),
        "lsa": lambda: LSARetriever(),
        "dense": dense,
        "hybrid": lambda: HybridRetriever(BM25Retriever(), dense()),
    }
    unknown = set(names) - set(available)
    if unknown:
        raise SystemExit(
            f"unknown retriever(s) {sorted(unknown)}; supported: {sorted(available)}"
        )
    return [available[name]() for name in names]


def run_retriever(retriever, documents, doc_ids, queries):
    """Index once, search every query. Returns (run, index_seconds, query_ms)."""
    start = time.time()
    retriever.index(documents)
    index_seconds = time.time() - start

    run = {}
    start = time.time()
    for query in queries:
        hits = retriever.search(query["text"], DEPTH)
        run[query["id"]] = [doc_ids[i] for i, _ in hits]
    query_ms = (time.time() - start) / max(len(queries), 1) * 1000
    return run, index_seconds, query_ms


def print_table(title, rows):
    print(f"\n=== {title} ===")
    header = f"{'retriever':<24} {'nDCG@10':>8} {'Recall@10':>10} {'MRR@10':>8} {'idx(s)':>8} {'q(ms)':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['retriever']:<24} {row['ndcg@10']:>8.4f} {row['recall@10']:>10.4f} "
            f"{row['mrr@10']:>8.4f} {row['index_seconds']:>8.1f} {row['query_ms']:>8.1f}"
        )


def evaluate_synthetic(retriever_names, model_name):
    corpus = json.loads((ROOT / "datasets" / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "datasets" / "queries.json").read_text(encoding="utf-8"))

    documents = [f"{d['title']}. {d['text']}" for d in corpus]
    doc_ids = [d["id"] for d in corpus]
    qrels = {q["id"]: {doc_id: 1 for doc_id in q["relevant"]} for q in queries}

    cache = embedding_cache_path("synthetic", model_name)
    rows, by_type = [], {}

    for retriever in make_retrievers(retriever_names, cache, model_name):
        run, index_seconds, query_ms = run_retriever(retriever, documents, doc_ids, queries)
        metrics = evaluate_run(run, qrels)
        rows.append({
            "retriever": retriever.name, **metrics,
            "index_seconds": round(index_seconds, 2), "query_ms": round(query_ms, 1),
        })

        # The breakdown is the point of this corpus: which method wins where.
        for query_type in sorted({q["type"] for q in queries}):
            subset = [q["id"] for q in queries if q["type"] == query_type]
            sub_run = {q: run[q] for q in subset}
            sub_qrels = {q: qrels[q] for q in subset}
            by_type.setdefault(query_type, {})[retriever.name] = evaluate_run(
                sub_run, sub_qrels
            )["ndcg@10"]

    print_table("Synthetic mixed-enterprise corpus (offline; NOT a benchmark)", rows)

    print("\n--- nDCG@10 by query type ---")
    names = [r["retriever"] for r in rows]
    print(f"{'query type':<22}" + "".join(f"{n:>26}" for n in names))
    for query_type, scores in sorted(by_type.items()):
        cells = "".join(f"{scores.get(n, 0.0):>26.4f}" for n in names)
        print(f"{query_type:<22}{cells}")

    return {"synthetic": {"overall": rows, "by_query_type": by_type}}


def evaluate_beir(dataset, retriever_names, model_name):
    corpus, queries, qrels = load_beir(dataset)
    documents = [
        (f"{d['title']}. {d['text']}" if d["title"] else d["text"]) for d in corpus
    ]
    doc_ids = [d["id"] for d in corpus]

    cache = embedding_cache_path(dataset, model_name)
    rows = []
    for retriever in make_retrievers(retriever_names, cache, model_name):
        run, index_seconds, query_ms = run_retriever(retriever, documents, doc_ids, queries)
        metrics = evaluate_run(run, qrels)
        rows.append({
            "retriever": retriever.name, **metrics,
            "index_seconds": round(index_seconds, 2), "query_ms": round(query_ms, 1),
        })

    print_table(
        f"BEIR / {dataset}  ({len(corpus)} docs, {len(queries)} queries, human qrels)",
        rows,
    )
    return rows


def split_timings(results):
    """Separate machine-dependent timings from machine-independent quality.

    Walks the nested result rows in place-safe fashion and returns two payloads
    with the same shape, so neither file has to know how the other is laid out.
    """
    quality = json.loads(json.dumps(results))
    timings = {}

    def strip(rows, path):
        kept = []
        for row in rows:
            timing = {key: row.pop(key) for key in TIMING_KEYS if key in row}
            if timing:
                timings.setdefault(path, {})[row["retriever"]] = timing
            kept.append(row)
        return kept

    if "synthetic" in quality:
        strip(quality["synthetic"]["overall"], "synthetic")
    for dataset, rows in quality.get("beir", {}).items():
        strip(rows, f"beir/{dataset}")

    return quality, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beir", default=None,
                        help="BEIR subset name, or 'all'; omit for the synthetic corpus")
    parser.add_argument("--retrievers", default="bm25,lsa,dense,hybrid")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    results = {"model": args.model, "retrievers": names}

    if args.beir:
        datasets = sorted(DATASETS) if args.beir == "all" else [args.beir]
        results["beir"] = {d: evaluate_beir(d, names, args.model) for d in datasets}
    else:
        results.update(evaluate_synthetic(names, args.model))

    if not args.no_save:
        quality, timings = split_timings(results)

        existing = {}
        if RESULTS_PATH.exists():
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        existing.update(quality)
        RESULTS_PATH.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(f"\nresults  -> {RESULTS_PATH.relative_to(ROOT)}")

        LATENCY_PATH.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
        print(f"timings  -> {LATENCY_PATH.relative_to(ROOT)} (not committed)")


if __name__ == "__main__":
    main()
