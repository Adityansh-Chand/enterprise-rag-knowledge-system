import numpy as np
from collections import Counter

from rag.embedder import tokenize


def cosine_similarity(a,b):

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    return float(np.dot(a,b)/denominator)


def keyword_score(query, text):

    q_tokens = tokenize(query)

    t_tokens = tokenize(text)

    overlap = Counter(q_tokens) & Counter(t_tokens)

    return sum(overlap.values())


class HybridRetriever:

    def __init__(self):

        self.texts = []

        self.vectors = []


    def add(self,text,vector):

        self.texts.append(text)

        self.vectors.append(vector)


    def search(self,query,query_vector,top_k=3):

        scores = []

        for text,vector in zip(self.texts,self.vectors):

            semantic = cosine_similarity(query_vector,vector)

            keyword = keyword_score(query,text)

            lexical = min(1.0, keyword / max(1, len(set(tokenize(query)))))

            combined = 0.65*semantic + 0.35*lexical

            scores.append((combined,text))


        ranked = sorted(

            scores,

            key=lambda x:x[0],

            reverse=True

        )

        return ranked[:top_k]
