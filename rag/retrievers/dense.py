"""Dense retrieval with a pretrained sentence-transformer bi-encoder.

Unlike LSA, this does not learn from the indexed corpus -- it arrives already
knowing that "why is checkout timing out" and "elevated p99 latency on the
payment gateway" are related, because it was pretrained on paraphrase data at a
scale no local corpus can match. That is its advantage.

Its weakness is the mirror image: rare exact identifiers (`ERR-4021`) carry
little pretrained meaning, so BM25 routinely beats it on those. The evaluation
harness reports both, per query type, rather than picking a winner up front.

Encoding is the expensive step, so document embeddings are cached to disk and
reused across harness runs.
"""
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

# bge models are trained with an asymmetric instruction: queries get a prefix,
# documents do not. Omitting it measurably degrades retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: Optional[Path] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else None
        self.batch_size = batch_size
        self._model = None
        self._matrix = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, documents: Sequence[str]) -> None:
        if self.cache_path and self.cache_path.exists():
            matrix = np.load(self.cache_path)
            if len(matrix) == len(documents):
                self._matrix = matrix
                return
            # A stale cache is worse than none -- fall through and re-encode.

        model = self._load_model()
        self._matrix = model.encode(
            list(documents),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, self._matrix)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")
        model = self._load_model()
        vector = model.encode(
            [QUERY_PREFIX + query], show_progress_bar=False, normalize_embeddings=True
        )[0]
        scores = self._matrix @ vector
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]
