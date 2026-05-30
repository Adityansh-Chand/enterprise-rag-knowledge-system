import hashlib
import re

import numpy as np

from config import EMBEDDING_PROVIDER


VECTOR_SIZE = 128
STOPWORDS = {
    "a", "an", "and", "are", "as", "can", "during", "for", "from", "how", "i",
    "is", "it", "of", "on", "or", "the", "to", "what", "with", "you", "your"
}


def normalize_token(token):
    token = token.lower()
    synonyms = {
        "vacation": "leave",
        "vacations": "leave",
        "remote": "remote",
        "remotely": "remote",
        "wfh": "remote",
    }
    if token in synonyms:
        return synonyms[token]

    for suffix in ("ing", "ed", "ly", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token


def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [normalize_token(token) for token in tokens if token not in STOPWORDS]


def _bucket(token):
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % VECTOR_SIZE


def local_embedding(text):
    vector = np.zeros(VECTOR_SIZE, dtype=float)
    for token in tokenize(text):
        vector[_bucket(token)] += 1.0

    norm = np.linalg.norm(vector)
    if norm:
        vector = vector / norm
    return vector


def embed(text):
    if EMBEDDING_PROVIDER == "local":
        return local_embedding(text)

    raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")
