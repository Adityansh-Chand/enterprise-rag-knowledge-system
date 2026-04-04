import numpy as np

def simple_embedding(text):

    return np.array([

        len(text),

        sum(ord(c) for c in text) % 997

    ])
