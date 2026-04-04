import numpy as np
from config import EMBEDDING_PROVIDER


def local_embedding(text):

    return np.array([
        len(text),
        sum(ord(c) for c in text) % 997,
        text.count(" ")
    ])


def embed(text):

    if EMBEDDING_PROVIDER == "local":
        return local_embedding(text)

    raise ValueError("Unknown embedding provider")
