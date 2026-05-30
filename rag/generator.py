def generate_answer(context, query):
    if not context:
        return "I could not find relevant information in the knowledge base."

    best_sentence = context.split("\n", 1)[0].strip()
    return f"Based on the knowledge base: {best_sentence}"


def build_response(context, results, query):

    answer = generate_answer(context, query)

    confidence = float(results[0][0]) if results else 0.0

    sources = [
        {
            "text": text,
            "score": float(score)
        }
        for score, text in results
    ]

    return {

        "answer": answer,
        "confidence": confidence,
        "sources": sources

    }
