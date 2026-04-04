import numpy as np

def cosine_similarity(a,b):

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


class VectorIndex:

    def __init__(self):

        self.texts = []

        self.vectors = []

    def add(self,text,vector):

        self.texts.append(text)

        self.vectors.append(vector)

    def search(self,query_vector,top_k=3):

        scores = [

            cosine_similarity(query_vector,v)

            for v in self.vectors

        ]

        ranked = sorted(

            zip(scores,self.texts),

            key=lambda x: x[0],

            reverse=True

        )

        return ranked[:top_k]
