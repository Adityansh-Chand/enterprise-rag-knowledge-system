"""Generate the evaluation set for answer generation, as opposed to retrieval.

Retrieval evaluation asks "did the right document come back". That question is
answered by `evaluation/harness.py` against human judgments. It says nothing
about the answer built on top of those documents, and the two can diverge
sharply: perfect retrieval followed by an answer that omits the remediation
step, or cites the wrong chunk, or confidently answers a question the corpus
cannot answer at all.

This file produces what is needed to measure that, and nothing here is a model's
opinion of its own output -- every judgment is checkable against the corpus.

Two parts:

**Answer facts.** Each topic carries the content units a correct answer has to
convey, as term groups. A fact counts as covered when every one of its terms
appears in the answer. Deliberately not "does the answer look right" -- that is
unfalsifiable. The generator asserts that every fact's terms really do appear in
the source document, so a fact can never be uncoverable by construction.

**Unanswerable queries.** Plausible, in-domain questions the corpus cannot
answer: an error code that does not exist, a clause number never written, an
endpoint that was never built. Each shares vocabulary with real documents
("remediation steps", "clause", "obligations", "pagination"), so retrieval
returns something confident every time. The correct behaviour is to decline.
These are the queries that make groundedness stop being a free pass: an answer
extracted from a retrieved chunk is perfectly grounded and still wrong.

The generator asserts each unanswerable query's distinguishing token appears
nowhere in the corpus, so "unanswerable" is a verified property rather than a
label.

Deterministic: no wall-clock, no randomness.

    python training/generate_generation_eval.py           # write the file
    python training/generate_generation_eval.py --check    # fail if it would differ
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.generation_metrics import terms  # noqa: E402
from training.generate_corpus import build  # noqa: E402

OUTPUT_PATH = ROOT / "datasets" / "generation_eval.json"

# --------------------------------------------------------------------------
# What a correct answer has to say, per topic.
#
# A term group is satisfied only when every term in it appears. Groups are drawn
# from the remediation and obligation content -- the part of a document that
# actually answers the question -- rather than from its restatement of the
# problem, because the restatement is what a query-overlap extractor picks up
# anyway and scoring it would measure nothing.
# --------------------------------------------------------------------------

ANSWER_FACTS = {
    "support:ERR-4021": [
        ["inspect", "response", "envelope"],
        ["merchant", "descriptor", "registered"],
        ["retry", "authorization", "once"],
        ["escalate", "issuer_unavailable"],
    ],
    "support:ERR-5503": [
        ["confirm", "connection", "pool", "saturation"],
        ["inspect", "slow", "query", "logs"],
        ["compare", "deployment", "marker"],
        ["roll", "back", "release"],
    ],
    "support:ERR-3310": [
        ["verify", "key", "identifier", "token", "header"],
        ["confirm", "rotation", "completed", "replicas"],
        ["invalidate", "cached", "public", "keys"],
    ],
    "support:ERR-2907": [
        ["check", "consumer", "health"],
        ["confirm", "partition", "assignment", "balanced"],
        ["scale", "consumer", "group"],
        ["retention", "window", "expires"],
    ],
    "support:ERR-6142": [
        ["identify", "long-running", "transactions"],
        ["confirm", "pool", "ceiling", "database", "maximum"],
        ["terminate", "idle", "sessions"],
    ],
    "support:ERR-7788": [
        ["renew", "certificate"],
        ["redistribute", "every", "replica"],
        ["confirm", "expiry", "monitor"],
    ],
    "finance:expense-travel": [
        ["sixty", "days", "trip"],
        ["itemised", "receipt"],
        ["economy", "fare", "cap", "prior", "written", "approval"],
        ["per", "diem", "published", "schedule"],
    ],
    "finance:net-payment-terms": [
        ["net", "sixty", "invoice", "issue", "date"],
        ["ten", "days", "two", "percent", "discount"],
        ["interest", "statutory", "rate"],
        ["collections", "ninety", "days"],
    ],
    "finance:procurement-threshold": [
        ["below", "ten", "thousand", "single", "department", "approval"],
        ["finance", "review", "competitive", "quote"],
        ["formal", "tender", "executive"],
    ],
    "finance:revenue-recognition": [
        ["recognised", "rateably", "committed", "service", "period"],
        ["unbundled", "performance", "obligations"],
        ["implementation", "fee", "milestone"],
    ],
    "finance:chargeback-handling": [
        ["cardholder", "disputes", "settled", "transaction"],
        ["compelling", "evidence", "issuer", "response", "window"],
        ["monitoring", "programme", "additional", "fees"],
    ],
    "legal:7.3": [
        ["aggregate", "liability", "fees", "paid", "twelve", "months"],
        ["death", "personal", "injury", "fraud"],
        ["indirect", "consequential", "losses", "excluded"],
    ],
    "legal:12.1": [
        ["register", "sub-processors"],
        ["thirty", "days", "notice"],
        ["controller", "object", "data", "protection", "grounds"],
    ],
    "legal:4.8": [
        ["initial", "term", "twenty-four", "months"],
        ["automatic", "renewal", "twelve", "month"],
        ["notice", "ninety", "days", "renewal", "date"],
    ],
    "legal:9.2": [
        ["transaction", "records", "retained", "seven", "years"],
        ["personal", "data", "deleted", "retention", "basis"],
        ["deletion", "evidenced", "audit", "register"],
    ],
    "product:POST /v2/invoices": [
        ["customer", "identifier", "line", "item"],
        ["idempotent", "idempotency-key", "header"],
        ["409", "duplicate", "key", "replayed"],
    ],
    "product:GET /v2/subscriptions": [
        ["paginated", "list", "subscriptions"],
        ["cursor", "parameter", "page"],
        ["status", "filter", "active", "cancelled"],
        ["page", "size", "defaults", "fifty"],
    ],
    "product:retry.max_attempts": [
        ["failed", "webhook", "delivery", "retried"],
        ["backoff", "exponential", "jitter"],
        ["parked", "deliveries", "replayed"],
    ],
    "product:index.refresh_interval": [
        ["newly", "written", "documents", "visible"],
        ["lowering", "improves", "freshness"],
        ["indexing", "throughput", "segment", "churn"],
    ],
}

# --------------------------------------------------------------------------
# Questions the corpus cannot answer.
#
# `absent` is the token that makes each one unanswerable, asserted to appear in
# no document. The rest of every query is deliberately ordinary corpus
# vocabulary, so retrieval returns a confident, plausible, wrong chunk.
# --------------------------------------------------------------------------

UNANSWERABLE = [
    ("ERR-9001 remediation steps", "err-9001"),
    ("ERR-1200 rollback procedure for the payments service", "err-1200"),
    ("clause 15.4 obligations", "15.4"),
    ("clause 22.9 termination for convenience", "22.9"),
    # Markers are normalised terms, so the leading slash of an API path is gone
    # by the time the assertion compares them -- see `generation_metrics.terms`.
    ("DELETE /v2/refunds request body", "v2/refunds"),
    ("GET /v2/disputes pagination cursor", "v2/disputes"),
    ("cryptocurrency settlement policy for customer payments", "cryptocurrency"),
    ("carbon reporting obligations under the framework agreement", "carbon"),
    ("parental leave entitlement for contractors", "parental"),
    ("on-call compensation rates for weekend shifts", "compensation"),
    ("queue.max_inflight configuration reference", "queue.max_inflight"),
    ("index.max_shards tuning guidance", "index.max_shards"),
]


def build_eval():
    documents, queries = build()

    text_by_topic = {}
    for doc in documents:
        if doc.get("canonical"):
            text_by_topic[doc["topic"]] = f"{doc['title']} {doc['text']}"

    corpus_terms = set()
    for doc in documents:
        corpus_terms.update(terms(f"{doc['title']} {doc['text']}"))

    problems = []

    # Every fact must be present in its own source document. Without this a
    # typo silently creates a fact nothing can ever cover, and the coverage
    # metric quietly reports a ceiling below 1.0 for no real reason.
    for topic, facts in sorted(ANSWER_FACTS.items()):
        source = text_by_topic.get(topic)
        if source is None:
            problems.append(f"{topic}: no canonical document")
            continue
        source_terms = terms(source)
        for fact in facts:
            missing = [term for term in fact if term not in source_terms]
            if missing:
                problems.append(f"{topic}: fact {fact} missing {missing} from source")

    for text, absent in UNANSWERABLE:
        if absent in corpus_terms:
            problems.append(
                f"unanswerable query {text!r} is answerable: {absent!r} is in the corpus"
            )
        if absent not in terms(text):
            problems.append(
                f"unanswerable query {text!r} does not contain its marker {absent!r}"
            )

    # Every answerable query needs facts, or it silently contributes nothing.
    topic_by_id = {doc["id"]: doc["topic"] for doc in documents}
    answerable = []
    for query in queries:
        topics = {topic_by_id[doc_id] for doc_id in query["relevant"]}
        topic = topics.pop() if len(topics) == 1 else None
        if topic not in ANSWER_FACTS:
            problems.append(f"{query['id']}: topic {topic} has no answer facts")
            continue
        answerable.append(
            {
                "id": query["id"],
                "text": query["text"],
                "type": query["type"],
                "topic": topic,
                "relevant": query["relevant"],
                "facts": ANSWER_FACTS[topic],
            }
        )

    if problems:
        raise AssertionError(
            "generation eval set is inconsistent with the corpus:\n  "
            + "\n  ".join(problems)
        )

    unanswerable = [
        {"id": f"u{index:03d}", "text": text, "type": "unanswerable", "absent": absent}
        for index, (text, absent) in enumerate(UNANSWERABLE, start=1)
    ]

    return {"answerable": answerable, "unanswerable": unanswerable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = build_eval()
    text = json.dumps(payload, indent=2) + "\n"

    if args.check:
        if not OUTPUT_PATH.exists():
            print("FAIL: generation_eval.json missing; run without --check first")
            return 1
        if OUTPUT_PATH.read_text(encoding="utf-8") != text:
            print("FAIL: regenerated generation eval set differs from the committed file")
            return 1
        print(
            f"OK: generation eval reproducible "
            f"({len(payload['answerable'])} answerable, "
            f"{len(payload['unanswerable'])} unanswerable)"
        )
        return 0

    OUTPUT_PATH.write_text(text, encoding="utf-8")
    facts = sum(len(q["facts"]) for q in payload["answerable"])
    print(
        f"wrote {len(payload['answerable'])} answerable queries "
        f"({facts} fact checks) and {len(payload['unanswerable'])} unanswerable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
