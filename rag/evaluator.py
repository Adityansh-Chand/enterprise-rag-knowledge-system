def recall_at_k(results, expected_text):

    retrieved = [r[1] for r in results]

    return int(expected_text in " ".join(retrieved))
