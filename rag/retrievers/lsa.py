"""Latent Semantic Analysis -- TF-IDF reduced by truncated SVD.

Genuinely semantic and genuinely fitted: terms that co-occur in similar contexts
collapse toward the same directions, so relations like `chargeback ~ dispute`
emerge from the corpus instead of being hand-written into a synonym table.

Its ceiling is real but modest. LSA learns only what the indexed corpus shows it,
so on a small corpus the latent space is thin and it will beat lexical retrieval
on paraphrase while falling well short of a pretrained bi-encoder. That is the
expected result, not a disappointing one -- reported rather than tuned away.
"""
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class LSARetriever:
    name = "lsa"

    def __init__(self, n_components: int = 128, min_df: int = 1):
        self.n_components = n_components
        self.min_df = min_df
        self._vectorizer = None
        self._svd = None
        self._matrix = None

    def index(self, documents: Sequence[str]) -> None:
        self._vectorizer = TfidfVectorizer(
            lowercase=True, sublinear_tf=True, min_df=self.min_df, stop_words="english"
        )
        tfidf = self._vectorizer.fit_transform(documents)

        # SVD cannot produce more components than the smaller matrix dimension.
        components = min(self.n_components, min(tfidf.shape) - 1)
        self._svd = TruncatedSVD(n_components=max(components, 2), random_state=42)
        self._matrix = normalize(self._svd.fit_transform(tfidf))

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")
        vector = normalize(self._svd.transform(self._vectorizer.transform([query])))
        scores = (self._matrix @ vector.T).ravel()
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]

    @property
    def explained_variance(self) -> float:
        return float(self._svd.explained_variance_ratio_.sum()) if self._svd else 0.0
