
def rerank(query, docs):
    return sorted(docs, key=lambda x: len(str(x)), reverse=True)
