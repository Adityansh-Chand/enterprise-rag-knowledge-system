def rerank(results):

    return sorted(

        results,

        key=lambda x: x[0],

        reverse=True

    )
