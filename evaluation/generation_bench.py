"""Evaluate the answer, not the ranking.

`evaluation/harness.py` measures retrieval: did the right document come back.
This measures everything after that, which is where a retrieval system fails a
user in ways nDCG cannot see -- an answer that omits the remediation step, an
answer quoted from the wrong document, an answer to a question the corpus cannot
answer at all.

The whole harness runs on the extractive path, offline, with no API key and no
spend. That is deliberate. The LLM path is one environment variable away and
scores on exactly the same metrics, so the harness that would judge a generative
system already exists and has been exercised; what it has not been given is a
vendor bill. Reported numbers come from the extractive path only, for the same
reason every other number in this repo does: a metric that depends on a model
version and a sampling temperature is not reproducible by a reviewer.

    python evaluation/generation_bench.py
    python evaluation/generation_bench.py --retriever dense
    python evaluation/generation_bench.py --retrievers bm25,dense --no-save
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.abstention import SIGNALS  # noqa: E402
from rag.generation_metrics import (  # noqa: E402
    attribution,
    context_utilisation,
    fact_coverage,
    groundedness,
    separation_auc,
    summarise,
)
from rag.generator import NO_ANSWER, generate_answer  # noqa: E402
from training.calibrate_abstention import build_pipeline, percentile  # noqa: E402

EVAL_PATH = ROOT / "datasets" / "generation_eval.json"
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "abstention.json"
RESULTS_PATH = ROOT / "evaluation" / "generation_results.json"

ANSWER_KEYS = ("groundedness", "fact_coverage", "attribution", "context_utilisation")

# Operating points to report the curve at. The served threshold stays the one
# fitted at the 5th percentile without ever looking at the unanswerable set --
# this curve is published so a reader can see what a different choice would cost,
# not so the best-looking point can be picked after the fact.
CURVE_PERCENTILES = (5.0, 10.0, 20.0, 30.0, 40.0)


def answer_with_provenance(pipeline, query_text, threshold, signal):
    """Produce an answer and everything needed to judge it."""
    results = pipeline.retrieve(query_text)
    chunks = [row[1] for row in results]
    scores = [row[0] for row in results]
    doc_ids = [row[3] for row in results]

    answer, mode = generate_answer(
        query_text, chunks, use_llm=False, scores=scores,
        signal=signal, threshold=threshold,
    )

    if mode == "extractive":
        from rag.generator import select_sentences

        sources = [
            doc_ids[rank]
            for _, rank, _ in select_sentences(query_text, chunks)
            if rank < len(doc_ids)
        ]
    else:
        sources = []

    return answer, mode, chunks, doc_ids, sources, scores


def evaluate_retriever(name, payload, artifact):
    pipeline = build_pipeline(name)
    signal = artifact["signal"]
    fitted = artifact.get("retrievers", {}).get(name)
    if fitted is None:
        raise SystemExit(
            f"no fitted abstention threshold for {name!r}; run "
            f"`python training/calibrate_abstention.py --retriever {name}` first"
        )
    threshold = fitted["thresholds"][signal]

    # --- answerable queries ------------------------------------------------
    rows, by_type = [], {}
    signal_values = {"answerable": {s: [] for s in SIGNALS},
                     "unanswerable": {s: [] for s in SIGNALS}}

    for query in payload["answerable"]:
        answer, mode, chunks, doc_ids, sources, scores = answer_with_provenance(
            pipeline, query["text"], threshold, signal
        )
        for signal_name, function in SIGNALS.items():
            signal_values["answerable"][signal_name].append(
                function(query["text"], chunks, scores)
            )

        row = {
            "id": query["id"],
            "type": query["type"],
            "mode": mode,
            "groundedness": groundedness(answer, chunks),
            "fact_coverage": fact_coverage(answer, query["facts"]),
            "attribution": attribution(sources, query["relevant"]),
            "context_utilisation": context_utilisation(sources, doc_ids),
        }
        rows.append(row)
        by_type.setdefault(query["type"], []).append(row)

    # --- unanswerable queries: the held-out test ---------------------------
    declined = 0
    for query in payload["unanswerable"]:
        answer, mode, chunks, doc_ids, sources, scores = answer_with_provenance(
            pipeline, query["text"], threshold, signal
        )
        for signal_name, function in SIGNALS.items():
            signal_values["unanswerable"][signal_name].append(
                function(query["text"], chunks, scores)
            )
        if answer == NO_ANSWER:
            declined += 1

    answered_when_it_should_not = len(payload["unanswerable"]) - declined
    answerable_declined = sum(1 for row in rows if row["mode"] == "abstained")

    return {
        "retriever": name,
        "signal": signal,
        "threshold": threshold,
        "n_answerable": len(rows),
        "n_unanswerable": len(payload["unanswerable"]),
        "answerable": summarise(rows, ANSWER_KEYS),
        "by_query_type": {
            query_type: summarise(subset, ANSWER_KEYS)
            for query_type, subset in sorted(by_type.items())
        },
        "abstention": {
            "unanswerable_declined": round(declined / len(payload["unanswerable"]), 4),
            "hallucination_rate": round(
                answered_when_it_should_not / len(payload["unanswerable"]), 4
            ),
            "answerable_declined": round(answerable_declined / len(rows), 4),
            "signal_auc": {
                signal_name: round(
                    separation_auc(
                        signal_values["answerable"][signal_name],
                        signal_values["unanswerable"][signal_name],
                    ),
                    4,
                )
                for signal_name in sorted(SIGNALS)
            },
            "curve": operating_curve(
                signal_values["answerable"][signal],
                signal_values["unanswerable"][signal],
            ),
        },
    }


def operating_curve(answerable_values, unanswerable_values):
    """What each percentile choice would cost and buy, on the served signal."""
    curve = []
    for q in CURVE_PERCENTILES:
        threshold = percentile(answerable_values, q)
        curve.append({
            "percentile": q,
            "threshold": round(threshold, 4),
            "answerable_declined": round(
                sum(1 for v in answerable_values if v < threshold)
                / len(answerable_values), 4),
            "unanswerable_declined": round(
                sum(1 for v in unanswerable_values if v < threshold)
                / len(unanswerable_values), 4),
        })
    return curve


def print_report(result):
    print(f"\n=== Generation quality: {result['retriever']} "
          f"({result['n_answerable']} answerable, "
          f"{result['n_unanswerable']} unanswerable) ===")

    overall = result["answerable"]
    print(f"{'metric':<22}{'overall':>10}")
    print("-" * 32)
    for key in ANSWER_KEYS:
        print(f"{key:<22}{overall[key]:>10.4f}")

    print("\n--- by query type ---")
    types = sorted(result["by_query_type"])
    print(f"{'metric':<22}" + "".join(f"{t:>22}" for t in types))
    for key in ANSWER_KEYS:
        cells = "".join(
            f"{result['by_query_type'][t][key]:>22.4f}" for t in types
        )
        print(f"{key:<22}{cells}")

    abstention = result["abstention"]
    print(f"\n--- abstention (signal: {result['signal']}, "
          f"threshold {result['threshold']}) ---")
    print(f"{'unanswerable declined':<32}{abstention['unanswerable_declined']:>10.4f}")
    print(f"{'hallucination rate':<32}{abstention['hallucination_rate']:>10.4f}")
    print(f"{'answerable wrongly declined':<32}{abstention['answerable_declined']:>10.4f}")
    print("\n--- signal separation (AUC, answerable vs unanswerable) ---")
    for name, value in sorted(abstention["signal_auc"].items()):
        print(f"{name:<32}{value:>10.4f}")

    print("\n--- what a different operating point would cost ---")
    print(f"{'percentile':<12}{'threshold':>12}{'answerable lost':>18}"
          f"{'unanswerable caught':>22}")
    for point in abstention["curve"]:
        print(f"{point['percentile']:<12}{point['threshold']:>12.4f}"
              f"{point['answerable_declined']:>18.4f}"
              f"{point['unanswerable_declined']:>22.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrievers", default="bm25")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    results = {}
    for name in names:
        result = evaluate_retriever(name, payload, artifact)
        print_report(result)
        results[name] = result

    if not args.no_save:
        existing = {}
        if RESULTS_PATH.exists():
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        existing.update(results)
        RESULTS_PATH.write_text(
            json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nresults -> {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
