"""Fit the threshold below which the system declines to answer.

The obvious way to set this is to take the answerable and unanswerable queries,
sweep the threshold, and keep whichever value maximises F1. It would also be
worthless: the unanswerable set is the only held-out test of abstention there
is, and a threshold chosen on it has already seen the answer.

So the threshold is fitted from **answerable queries only**, by the same
reasoning the incident repo uses for alert budgets -- you rarely have labelled
negatives in production, but you always have the distribution of the positives.
Take the signal value each answerable query achieves and put the threshold at a
low percentile of that distribution. The percentile is the operating choice; it
says "accept declining this share of answerable questions". What that choice
buys on unanswerable questions is then reported honestly by
`evaluation/generation_bench.py`, which is the first time those queries are
looked at.

Thresholds are fitted per retriever because the served signal is a raw retrieval
score: BM25 emits unbounded term weights and a bi-encoder emits normalised
cosines, and one number cannot mean the same thing in both.

    python training/calibrate_abstention.py --retriever bm25
    python training/calibrate_abstention.py --retriever dense
    python training/calibrate_abstention.py --retriever bm25 --verify
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.abstention import SIGNALS  # noqa: E402
from rag.beir_data import embedding_cache_path  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retrievers.dense import DEFAULT_MODEL  # noqa: E402

EVAL_PATH = ROOT / "datasets" / "generation_eval.json"
CORPUS_PATH = ROOT / "datasets" / "corpus.json"
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "abstention.json"

# The share of answerable questions we accept declining. Low, because declining
# a question the corpus can answer is the more visible failure -- the user knows
# the answer exists. Configurable because it is an operating choice, not a fact
# about the data.
DEFAULT_PERCENTILE = 5.0


def percentile(values, q):
    """Linear-interpolated percentile. Avoids a numpy import for one number."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def build_pipeline(retriever_name):
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    cache = (
        embedding_cache_path("synthetic", DEFAULT_MODEL)
        if retriever_name in ("dense", "hybrid", "router")
        else None
    )
    pipeline = RAGPipeline(retriever_name=retriever_name, cache_path=cache)
    for document in corpus:
        # Passages, not re-chunked: doc ids stay identical to the corpus ids, so
        # attribution can be checked against the relevance judgments directly.
        pipeline.ingest_passage(
            f"{document['title']}. {document['text']}",
            title=document["title"],
            doc_id=document["id"],
        )
    pipeline.build_index()
    return pipeline


def signal_values(pipeline, queries):
    """Every candidate signal, on every query, from one retrieval pass each."""
    values = {name: [] for name in SIGNALS}
    for query in queries:
        results = pipeline.retrieve(query["text"])
        chunks = [row[1] for row in results]
        scores = [row[0] for row in results]
        for name, function in SIGNALS.items():
            values[name].append(function(query["text"], chunks, scores))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--signal", default=None,
                        help="which signal the served pipeline uses; "
                             "omit to keep the committed choice")
    parser.add_argument("--abstain", dest="abstain", action="store_true", default=None,
                        help="serve abstention for this retriever")
    parser.add_argument("--no-abstain", dest="abstain", action="store_false",
                        help="fit and record the threshold, but do not serve it")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    pipeline = build_pipeline(args.retriever)
    values = signal_values(pipeline, payload["answerable"])

    existing = {}
    if ARTIFACT_PATH.exists():
        try:
            existing = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}

    previous = existing.get("retrievers", {}).get(args.retriever, {})
    fitted = {
        # Whether the fitted threshold is actually used in serving. Fitting and
        # serving are separated because a signal can be measurable and still be
        # the wrong thing to act on -- see ADR-005 on BM25, where the score is
        # not comparable across queries and the threshold lands on the queries
        # BM25 answers best.
        "abstain": previous.get("abstain", True) if args.abstain is None
        else args.abstain,
        "thresholds": {
            name: round(percentile(series, args.percentile), 4)
            for name, series in sorted(values.items())
        },
        "distribution": {
            name: {
                "min": round(min(series), 4),
                "median": round(percentile(series, 50), 4),
                "max": round(max(series), 4),
            }
            for name, series in sorted(values.items())
        },
    }

    retrievers = dict(existing.get("retrievers", {}))
    retrievers[args.retriever] = fitted

    artifact = {
        "signal": args.signal or existing.get("signal", "top_score"),
        "percentile": args.percentile,
        "fitted_on": "answerable queries only",
        "n_answerable": len(payload["answerable"]),
        "retrievers": {name: retrievers[name] for name in sorted(retrievers)},
    }
    text = json.dumps(artifact, indent=2) + "\n"

    if args.verify:
        if not ARTIFACT_PATH.exists():
            print("FAIL: abstention.json missing; run without --verify first")
            return 1
        if ARTIFACT_PATH.read_text(encoding="utf-8") != text:
            print("FAIL: refitted abstention thresholds differ from the committed file")
            return 1
        print(f"OK: abstention thresholds reproducible for {args.retriever}")
        return 0

    ARTIFACT_PATH.write_text(text, encoding="utf-8")
    print(f"{args.retriever}: fitted at the {args.percentile}th percentile of "
          f"{len(payload['answerable'])} answerable queries")
    for name in sorted(fitted["thresholds"]):
        stats = fitted["distribution"][name]
        print(f"  {name:<10} threshold {fitted['thresholds'][name]:<9} "
              f"min {stats['min']:<9} median {stats['median']:<9} max {stats['max']}")
    print(f"  served signal: {artifact['signal']}")
    print(f"artifact -> {ARTIFACT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
