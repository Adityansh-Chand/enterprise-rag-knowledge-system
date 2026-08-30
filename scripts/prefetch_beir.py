"""Download BEIR subsets and pre-compute dense embeddings into the local cache.

Run once before the evaluation harness. Embeddings are cached to disk so repeated
harness runs are free. Nothing here is committed except this script -- the cache
directory is gitignored.

    python scripts/prefetch_beir.py                # all configured datasets
    python scripts/prefetch_beir.py --dataset fiqa # one dataset
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.beir_data import CACHE_DIR, DATASETS, embedding_cache_path, load_beir  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    names = [args.dataset] if args.dataset else sorted(DATASETS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = None
    for name in names:
        corpus, queries, qrels = load_beir(name)
        print(f"[{name}] docs={len(corpus)} queries={len(queries)} qrels={len(qrels)}", flush=True)

        out = embedding_cache_path(name, args.model)
        if out.exists():
            print(f"[{name}] embeddings already cached at {out.name}", flush=True)
            continue

        if model is None:
            print(f"loading {args.model} ...", flush=True)
            model = SentenceTransformer(args.model)

        texts = [d["text"] for d in corpus]
        start = time.time()
        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        elapsed = time.time() - start
        np.save(out, vectors.astype("float32"))
        print(
            f"[{name}] encoded {len(texts)} docs in {elapsed/60:.1f} min "
            f"({len(texts)/elapsed:.1f} docs/s) -> {out.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
