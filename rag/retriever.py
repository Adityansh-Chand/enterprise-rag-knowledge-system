import numpy as np
from collections import Counter


def cosine_similarity(a,b):

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def keyword_score(query, text):

    q_tokens = query.lower().split()

    t_tokens = text.lower().split()

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

            combined = 0.7*semantic + 0.3*(keyword>0)

            scores.append((combined,text))


        ranked = sorted(

            scores,

            key=lambda x:x[0],

            reverse=True

        )

        return ranked[:top_k]
