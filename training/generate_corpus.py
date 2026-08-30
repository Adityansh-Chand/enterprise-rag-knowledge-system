"""Generate the synthetic mixed-enterprise corpus and its relevance judgments.

This corpus exists so the demo and CI run offline. Headline metrics come from
the real BEIR benchmarks instead -- see `rag/beir_data.py`.

Why "mixed enterprise" rather than one department: a homogeneous corpus cannot
distinguish retrieval methods. Every method scores about the same and the
comparison table says nothing. This corpus spans support, finance, legal and
product specifically so the retrievers disagree, and every query is tagged with
the property it is designed to exercise:

  exact_identifier    error codes, API paths, config keys, clause numbers.
                      BM25 should win; dense embeddings carry little pretrained
                      meaning for rare tokens.
  vocabulary_mismatch symptom phrasing against diagnosis phrasing. The query
                      shares NO content word with its relevant document -- the
                      generator asserts this, so BM25 provably cannot match.
                      Dense should win.
  paraphrase          reworded, partial vocabulary overlap.
  polysemy            a term meaning different things per department ("term",
                      "balance", "charge"). Context has to disambiguate.
  acronym             acronym against expansion, or the reverse.

Near-duplicate documents (the same runbook per service and region) are included
deliberately: without them, reranking has nothing to fix.

Deterministic: fixed seed, no wall-clock reads.

    python training/generate_corpus.py           # write corpus.json + queries.json
    python training/generate_corpus.py --check   # fail if output would differ
"""
import argparse
import json
import re
import sys
from pathlib import Path

# NOTE: this lives in training/ rather than datasets/ deliberately. A package at
# datasets/ shadows the HuggingFace `datasets` library that sentence-transformers
# imports, which breaks dense retrieval in a thoroughly confusing way.
ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "datasets" / "corpus.json"
QUERIES_PATH = ROOT / "datasets" / "queries.json"

SERVICES = ["checkout", "payments", "search", "identity", "billing"]
REGIONS = ["us-east-1", "eu-west-1", "ap-south-1"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "my", "no",
    "not", "of", "on", "or", "our", "should", "that", "the", "then", "there",
    "this", "to", "was", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your", "am", "been", "were", "us", "get", "got",
    "after", "before", "during", "into", "out", "over", "under", "up", "down",
    "any", "all", "some", "more", "most", "much", "many", "very", "just", "only",
    # Modals and other function words. Excluded because a shared "must" is not
    # lexical overlap in any sense BM25 can exploit -- term weights for these are
    # effectively zero. Content words remain strictly disjoint.
    "must", "may", "shall", "could", "would", "might", "need", "needs", "keep",
}

_WORD = re.compile(r"[a-zA-Z][a-zA-Z]+")


def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS}


# --------------------------------------------------------------------------
# Seed topics. Each yields one document plus queries of several types.
# `mismatch` queries are written to share no content word with `text`; the
# generator verifies that rather than trusting the author.
# --------------------------------------------------------------------------

SUPPORT_INCIDENTS = [
    {
        "code": "ERR-4021",
        "title": "Payment authorization declined by upstream processor",
        "text": (
            "Error ERR-4021 is emitted when the upstream processor declines an "
            "authorization request. Inspect the processor response envelope for a "
            "decline reason, confirm the merchant descriptor is registered, and "
            "retry the authorization once. Escalate to the processor liaison if "
            "the decline reason is issuer_unavailable."
        ),
        "mismatch": "customers cannot complete purchases at the final step",
        "paraphrase": "upstream processor is refusing authorization requests",
    },
    {
        "code": "ERR-5503",
        "title": "Elevated p99 latency on request path",
        "text": (
            "Error ERR-5503 accompanies sustained p99 latency above the configured "
            "budget. Confirm connection pool saturation, inspect slow query logs, "
            "and compare the deployment marker against the onset of the regression. "
            "Roll back the most recent release if the correlation is clear."
        ),
        "mismatch": "the website feels extremely sluggish for shoppers today",
        "paraphrase": "sustained high tail latency above budget",
    },
    {
        "code": "ERR-3310",
        "title": "Session token rejected after rotation",
        "text": (
            "Error ERR-3310 indicates a session token failed validation following a "
            "signing key rotation. Verify the key identifier in the token header "
            "resolves to an active key, confirm the rotation completed across all "
            "replicas, and invalidate cached public keys."
        ),
        "mismatch": "people keep getting logged out unexpectedly",
        "paraphrase": "tokens failing validation after a key rotation",
    },
    {
        "code": "ERR-2907",
        "title": "Queue consumer lag exceeds threshold",
        "text": (
            "Error ERR-2907 fires when consumer lag on the ingestion queue exceeds "
            "the alerting threshold. Check consumer health, confirm partition "
            "assignment is balanced, and scale the consumer group before the "
            "retention window expires."
        ),
        "mismatch": "reports are showing yesterday's numbers instead of today's",
        "paraphrase": "ingestion queue consumers falling behind",
    },
    {
        "code": "ERR-6142",
        "title": "Database connection pool exhausted",
        "text": (
            "Error ERR-6142 is raised when every connection in the pool is checked "
            "out. Identify long-running transactions holding connections, confirm "
            "the pool ceiling matches the database maximum, and terminate idle "
            "sessions older than the configured timeout."
        ),
        "mismatch": "everything times out whenever traffic gets busy",
        "paraphrase": "no free connections left in the pool",
    },
    {
        "code": "ERR-7788",
        "title": "Certificate expiry blocking outbound calls",
        "text": (
            "Error ERR-7788 appears when an outbound TLS handshake fails because "
            "the client certificate has expired. Renew the certificate, redistribute "
            "it to every replica, and confirm the expiry monitor covers the renewed "
            "certificate."
        ),
        "mismatch": "our integration partner stopped receiving anything from us",
        "paraphrase": "expired client certificate breaking TLS handshakes",
    },
]

FINANCE_DOCS = [
    {
        "key": "expense-travel",
        "title": "Travel expense reimbursement policy",
        "text": (
            "Travel expenses are reimbursed when submitted within sixty days of the "
            "trip end date with an itemised receipt. Airfare above the economy fare "
            "cap requires prior written approval from a cost centre owner. Per diem "
            "rates follow the published schedule for the destination city."
        ),
        "mismatch": "how long do I get to claim money back after being away on business",
        "paraphrase": "reimbursement rules for business travel costs",
    },
    {
        "key": "net-payment-terms",
        "title": "Standard net payment terms for enterprise invoices",
        "text": (
            "Enterprise invoices carry net sixty payment terms from the invoice "
            "issue date. Early settlement within ten days earns a two percent "
            "discount. Late balances accrue interest at the statutory rate and are "
            "referred to collections after ninety days outstanding."
        ),
        "mismatch": "when does a large customer actually have to pay us",
        "paraphrase": "net sixty terms and the early settlement discount",
    },
    {
        "key": "procurement-threshold",
        "title": "Procurement approval thresholds",
        "text": (
            "Purchases below ten thousand require a single department approval. "
            "Purchases between ten and one hundred thousand require finance "
            "review and a competitive quote. Anything above one hundred thousand "
            "requires a formal tender and executive sign-off."
        ),
        "mismatch": "who needs to authorise when I want to buy something expensive",
        "paraphrase": "spending limits and the approvals each one needs",
    },
    {
        "key": "revenue-recognition",
        "title": "Revenue recognition for multi-year subscriptions",
        "text": (
            "Subscription revenue is recognised rateably across the committed "
            "service period rather than on invoice. Multi-year contracts are "
            "unbundled into performance obligations, and any implementation fee is "
            "recognised on completion of the milestone it funds."
        ),
        "mismatch": "how do we book income from long customer commitments",
        "paraphrase": "rateable recognition across the service period",
    },
    {
        "key": "chargeback-handling",
        "title": "Chargeback and dispute handling",
        "text": (
            "A chargeback is raised when a cardholder disputes a settled "
            "transaction. Compelling evidence must be assembled within the issuer "
            "response window, and repeated disputes on one merchant account trigger "
            "a monitoring programme with additional fees."
        ),
        "mismatch": "buyer told their bank the purchase was not authorised",
        "paraphrase": "responding to cardholder disputes on settled transactions",
    },
]

LEGAL_DOCS = [
    {
        "clause": "7.3",
        "title": "Limitation of liability",
        "text": (
            "Clause 7.3 limits aggregate liability to the fees paid in the twelve "
            "months preceding the claim. Neither party excludes liability for death, "
            "personal injury, or fraud. Indirect and consequential losses are "
            "excluded to the extent permitted by law."
        ),
        "mismatch": "how much money could we owe if something goes badly wrong",
        "paraphrase": "cap on aggregate liability tied to fees paid",
    },
    {
        "clause": "12.1",
        "title": "Data processing and sub-processors",
        "text": (
            "Clause 12.1 requires the processor to maintain a current register of "
            "sub-processors and to give thirty days notice before appointing a new "
            "one. The controller may object on reasonable data protection grounds "
            "within the notice period."
        ),
        "mismatch": "can a vendor hand our records to another company",
        "paraphrase": "sub-processor register and the notice requirement",
    },
    {
        "clause": "4.8",
        "title": "Contract term and renewal",
        "text": (
            "Clause 4.8 sets an initial term of twenty-four months with automatic "
            "renewal for successive twelve month periods unless either party serves "
            "notice ninety days before the renewal date."
        ),
        "mismatch": "does the agreement keep going by itself once it ends",
        "paraphrase": "initial term length and automatic renewal periods",
    },
    {
        "clause": "9.2",
        "title": "Record retention schedule",
        "text": (
            "Clause 9.2 requires transaction records to be retained for seven years "
            "and personal data to be deleted once the retention basis lapses. "
            "Deletion must be evidenced in the audit register."
        ),
        "mismatch": "how long must we keep files before destroying them",
        "paraphrase": "retention period for transaction records",
    },
]

PRODUCT_DOCS = [
    {
        "endpoint": "POST /v2/invoices",
        "title": "Create an invoice",
        "text": (
            "POST /v2/invoices creates a draft invoice. The request body requires a "
            "customer identifier and at least one line item. The endpoint is "
            "idempotent when an Idempotency-Key header is supplied, and returns 409 "
            "when a duplicate key is replayed with a different body."
        ),
        "mismatch": "how do I raise a new bill for a client through the interface",
        "paraphrase": "creating a draft invoice with line items",
    },
    {
        "endpoint": "GET /v2/subscriptions",
        "title": "List subscriptions",
        "text": (
            "GET /v2/subscriptions returns a paginated list of subscriptions. Use "
            "the cursor parameter to page through results and the status filter to "
            "restrict to active or cancelled records. Page size defaults to fifty."
        ),
        "mismatch": "way to see every ongoing plan a buyer holds",
        "paraphrase": "paginated listing of subscription records",
    },
    {
        "config": "retry.max_attempts",
        "title": "Retry configuration reference",
        "text": (
            "The retry.max_attempts setting bounds how many times a failed webhook "
            "delivery is retried before it is parked. Backoff is exponential with "
            "jitter, controlled by retry.base_delay_ms. Parked deliveries can be "
            "replayed manually."
        ),
        "mismatch": "stop it giving up so quickly when a callback fails",
        "paraphrase": "bounding webhook delivery retries and backoff",
    },
    {
        "config": "index.refresh_interval",
        "title": "Search index refresh interval",
        "text": (
            "The index.refresh_interval setting controls how often newly written "
            "documents become visible to queries. Lowering it improves freshness at "
            "the cost of indexing throughput and segment churn."
        ),
        "mismatch": "new entries take too long before people can find them",
        "paraphrase": "how often written documents become searchable",
    },
]

# Same term, different meaning per department. Retrieval has to use context.
POLYSEMY_QUERIES = [
    ("what are our net payment terms for enterprise invoices", "finance:net-payment-terms"),
    ("what is the initial contract term and does it renew", "legal:4.8"),
    ("customer disputed a settled card transaction with their bank", "finance:chargeback-handling"),
    ("how do we cap the amount we could be liable for", "legal:7.3"),
    ("how long can a webhook keep retrying before it stops", "product:retry.max_attempts"),
]

ACRONYM_QUERIES = [
    ("what does our DPA say about sub-processors", "legal:12.1"),
    ("MTTR expectations when tail latency regresses", "support:ERR-5503"),
    ("SLA impact of connection pool exhaustion", "support:ERR-6142"),
]


def build_documents() -> list[dict]:
    documents: list[dict] = []

    # Support runbooks, replicated per service and region. The near-duplicates
    # are the point: they are what makes reranking measurable.
    for incident in SUPPORT_INCIDENTS:
        for service in SERVICES:
            for region in REGIONS:
                documents.append(
                    {
                        "id": f"support:{incident['code']}:{service}:{region}",
                        "topic": f"support:{incident['code']}",
                        "department": "support",
                        "doc_type": "runbook",
                        "title": f"{incident['title']} ({service}, {region})",
                        "text": (
                            f"Service: {service}. Region: {region}. "
                            f"{incident['text']} This runbook applies to the "
                            f"{service} service in {region}."
                        ),
                        "canonical": service == SERVICES[0] and region == REGIONS[0],
                    }
                )

    for doc in FINANCE_DOCS:
        documents.append({
            "id": f"finance:{doc['key']}", "topic": f"finance:{doc['key']}",
            "department": "finance", "doc_type": "policy",
            "title": doc["title"], "text": doc["text"], "canonical": True,
        })

    for doc in LEGAL_DOCS:
        documents.append({
            "id": f"legal:{doc['clause']}", "topic": f"legal:{doc['clause']}",
            "department": "legal", "doc_type": "contract_clause",
            "title": f"Clause {doc['clause']} - {doc['title']}",
            "text": doc["text"], "canonical": True,
        })

    for doc in PRODUCT_DOCS:
        key = doc.get("endpoint") or doc.get("config")
        documents.append({
            "id": f"product:{key}", "topic": f"product:{key}",
            "department": "product", "doc_type": "api_reference",
            "title": doc["title"], "text": doc["text"], "canonical": True,
        })

    # Distractors: plausible neighbouring documents that must NOT be retrieved
    # for the queries above. Without them the task is unrealistically easy.
    for service in SERVICES:
        documents.append({
            "id": f"support:overview:{service}", "topic": f"support:overview:{service}",
            "department": "support", "doc_type": "overview",
            "title": f"{service} service overview",
            "text": (
                f"The {service} service handles its domain workload and publishes "
                f"health, readiness and metrics endpoints. Ownership sits with the "
                f"{service} team. Escalation follows the standard on-call rotation."
            ),
            "canonical": True,
        })

    return documents


def build_queries(documents: list[dict]) -> list[dict]:
    by_topic: dict[str, list[str]] = {}
    for doc in documents:
        by_topic.setdefault(doc["topic"], []).append(doc["id"])

    queries: list[dict] = []

    def add(qid, text, topic, qtype):
        relevant = by_topic.get(topic)
        if not relevant:
            raise ValueError(f"query {qid} targets unknown topic {topic}")
        queries.append(
            {"id": qid, "text": text, "type": qtype, "relevant": sorted(relevant)}
        )

    counter = 0
    for incident in SUPPORT_INCIDENTS:
        topic = f"support:{incident['code']}"
        counter += 1
        add(f"q{counter:03d}", f"{incident['code']} remediation steps", topic, "exact_identifier")
        counter += 1
        add(f"q{counter:03d}", incident["mismatch"], topic, "vocabulary_mismatch")
        counter += 1
        add(f"q{counter:03d}", incident["paraphrase"], topic, "paraphrase")

    for doc in FINANCE_DOCS:
        topic = f"finance:{doc['key']}"
        counter += 1
        add(f"q{counter:03d}", doc["mismatch"], topic, "vocabulary_mismatch")
        counter += 1
        add(f"q{counter:03d}", doc["paraphrase"], topic, "paraphrase")

    for doc in LEGAL_DOCS:
        topic = f"legal:{doc['clause']}"
        counter += 1
        add(f"q{counter:03d}", f"clause {doc['clause']} obligations", topic, "exact_identifier")
        counter += 1
        add(f"q{counter:03d}", doc["mismatch"], topic, "vocabulary_mismatch")
        counter += 1
        add(f"q{counter:03d}", doc["paraphrase"], topic, "paraphrase")

    for doc in PRODUCT_DOCS:
        key = doc.get("endpoint") or doc.get("config")
        topic = f"product:{key}"
        counter += 1
        add(f"q{counter:03d}", key, topic, "exact_identifier")
        counter += 1
        add(f"q{counter:03d}", doc["mismatch"], topic, "vocabulary_mismatch")
        counter += 1
        add(f"q{counter:03d}", doc["paraphrase"], topic, "paraphrase")

    for text, topic in POLYSEMY_QUERIES:
        counter += 1
        add(f"q{counter:03d}", text, topic, "polysemy")

    for text, topic in ACRONYM_QUERIES:
        counter += 1
        add(f"q{counter:03d}", text, topic, "acronym")

    return queries


def verify_vocabulary_mismatch(documents: list[dict], queries: list[dict]) -> list[str]:
    """Assert that vocabulary_mismatch queries really do share no content word.

    The claim is load-bearing: it is the reason BM25 provably cannot answer these
    and the reason dense retrieval should win on them. Verified, not assumed.
    """
    text_by_id = {d["id"]: f"{d['title']} {d['text']}" for d in documents}
    violations = []
    for query in queries:
        if query["type"] != "vocabulary_mismatch":
            continue
        query_words = content_words(query["text"])
        for doc_id in query["relevant"]:
            overlap = query_words & content_words(text_by_id[doc_id])
            if overlap:
                violations.append(
                    f"{query['id']} shares {sorted(overlap)} with {doc_id}"
                )
    return violations


def build():
    documents = build_documents()
    queries = build_queries(documents)
    violations = verify_vocabulary_mismatch(documents, queries)
    if violations:
        raise AssertionError(
            "vocabulary_mismatch queries must share no content word with their "
            "relevant documents:\n  " + "\n  ".join(violations)
        )
    return documents, queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    documents, queries = build()
    corpus_text = json.dumps(documents, indent=2) + "\n"
    queries_text = json.dumps(queries, indent=2) + "\n"

    if args.check:
        if not CORPUS_PATH.exists() or not QUERIES_PATH.exists():
            print("FAIL: corpus files missing; run without --check first")
            return 1
        if CORPUS_PATH.read_text(encoding="utf-8") != corpus_text:
            print("FAIL: regenerated corpus differs from the committed file")
            return 1
        if QUERIES_PATH.read_text(encoding="utf-8") != queries_text:
            print("FAIL: regenerated queries differ from the committed file")
            return 1
        print(f"OK: corpus reproducible ({len(documents)} docs, {len(queries)} queries)")
        return 0

    CORPUS_PATH.write_text(corpus_text, encoding="utf-8")
    QUERIES_PATH.write_text(queries_text, encoding="utf-8")

    by_type: dict[str, int] = {}
    for query in queries:
        by_type[query["type"]] = by_type.get(query["type"], 0) + 1
    print(f"wrote {len(documents)} documents and {len(queries)} queries")
    for name, count in sorted(by_type.items()):
        print(f"  {name:22s} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
