def precision_at_k(results, expected):

    retrieved = [r[1] for r in results]

    return int(expected in " ".join(retrieved))
