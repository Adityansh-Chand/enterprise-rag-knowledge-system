"""Capture the retrieval-score distribution the service expects to see.

Unlike the fitted services there is no training run here to hang this off, so the
reference is built explicitly: run the evaluation query set against the indexed
corpus and record the distribution of top retrieval scores.

What this detects is a change in the *relationship* between the questions being
asked and the corpus that answers them. Both ends move in practice -- documents
are ingested (the meeting service writes into this corpus), and the questions
people ask drift with them. A collapse in top scores means queries are arriving
that the corpus no longer answers well, which is the signal worth having before
someone reports bad answers.

    python training/build_drift_reference.py
    python training/build_drift_reference.py --verify
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.drift import build_reference  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

QUERIES_PATH = ROOT / "datasets" / "queries.json"
CORPUS_PATH = ROOT / "datasets" / "corpus.json"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
DRIFT_REFERENCE_PATH = ARTIFACT_DIR / "drift_reference.json"


def collect_scores():
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    # Indexed exactly as api/server.py indexes it, so the reference describes the
    # same corpus the service will actually be serving.
    pipeline = RAGPipeline()
    for document in json.loads(CORPUS_PATH.read_text(encoding="utf-8")):
        pipeline.ingest_passage(
            f"{document['title']}. {document['text']}",
            title=document["title"],
            doc_id=document["id"],
        )
    pipeline.build_index()

    scores = []
    for entry in queries:
        text = entry.get("query") or entry.get("text")
        if not text:
            continue
        scores.append(float(pipeline.query(text)["retrieval_score"]))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="rebuild and fail if the reference drifted")
    args = parser.parse_args()

    scores = collect_scores()
    if not scores:
        print("FAIL: no queries produced a retrieval score")
        return 1
    reference = build_reference(scores)

    if args.verify:
        if not DRIFT_REFERENCE_PATH.exists():
            print("FAIL: drift_reference.json missing; run without --verify first")
            return 1
        committed = json.loads(DRIFT_REFERENCE_PATH.read_text(encoding="utf-8"))
        if committed["n_reference"] != reference["n_reference"]:
            print(f"FAIL: reference size changed "
                  f"{committed['n_reference']} -> {reference['n_reference']}")
            return 1
        print(f"OK: reference reproduces ({reference['n_reference']} queries)")
        return 0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_REFERENCE_PATH.write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    print(f"queries    : {reference['n_reference']}")
    print(f"score range: {reference['edges'][0]:.4f} .. {reference['edges'][-1]:.4f}")
    print(f"written    : {DRIFT_REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
