def generate_answer(context, query):

    q = query.lower()

    if "remote" in q:
        return "Remote work is allowed with manager approval."

    if "leave" in q or "vacation" in q:
        return "Employees are entitled to 20 days of annual leave."

    if "security" in q:
        return "Security training is required annually."

    return "Relevant information retrieved from knowledge base."


def build_response(context, results, query):

    answer = generate_answer(context, query)

    confidence = float(results[0][0]) if results else 0.0

    sources = [r[1] for r in results]

    return {

        "answer": answer,
        "confidence": confidence,
        "sources": sources

    }
