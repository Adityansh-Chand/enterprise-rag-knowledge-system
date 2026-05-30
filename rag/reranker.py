from rag.embedder import tokenize


def rerank(query, docs):
    query_terms = set(tokenize(query))

    def score(doc):
        base_score, text = doc
        doc_terms = set(tokenize(text))
        overlap = len(query_terms & doc_terms)
        coverage = overlap / max(1, len(query_terms))
        return (base_score * 0.8) + (coverage * 0.2)

    return sorted(docs, key=score, reverse=True)
