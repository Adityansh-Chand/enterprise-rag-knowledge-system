"""Loading and caching for the BEIR evaluation track.

These are real, publicly released IR benchmarks with human relevance judgments --
as opposed to the synthetic corpus in `datasets/`, which exists so the demo runs
offline. Metrics reported against these are comparable to published numbers;
metrics against the synthetic corpus are not.

Source: the BeIR collection on the HuggingFace hub, license cc-by-sa-4.0.
The official BEIR mirror (public.ukp.informatik.tu-darmstadt.de) was unusably
slow when this was written, so the hub is the download path.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "embeddings"

# Three subsets, deliberately chosen because they disagree with each other.
# fiqa is the case where dense retrieval beats lexical badly; scifact and
# nfcorpus are cases where BM25 stays competitive. Reporting only one would
# produce a misleadingly tidy conclusion.
DATASETS = {
    "nfcorpus": "BeIR/nfcorpus",
    "scifact": "BeIR/scifact",
    "fiqa": "BeIR/fiqa",
}


def embedding_cache_path(dataset: str, model_name: str) -> Path:
    slug = model_name.replace("/", "__")
    return CACHE_DIR / f"{dataset}__{slug}.npy"


def load_beir(name: str):
    """Return (corpus, queries, qrels) for a BEIR subset.

    corpus: list of {"id", "title", "text"} in a fixed order
    queries: list of {"id", "text"} restricted to those with test judgments
    qrels: dict query_id -> {doc_id: relevance}
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")

    from datasets import load_dataset

    repo = DATASETS[name]
    raw_corpus = load_dataset(repo, "corpus", split="corpus")
    raw_queries = load_dataset(repo, "queries", split="queries")
    raw_qrels = load_dataset(f"{repo}-qrels", split="test")

    qrels: dict[str, dict[str, int]] = {}
    for row in raw_qrels:
        score = int(row["score"])
        if score <= 0:
            continue
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score

    corpus = [
        {"id": str(r["_id"]), "title": r["title"] or "", "text": r["text"] or ""}
        for r in raw_corpus
    ]
    queries = [
        {"id": str(r["_id"]), "text": r["text"] or ""}
        for r in raw_queries
        if str(r["_id"]) in qrels
    ]
    return corpus, queries, qrels
