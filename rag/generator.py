def build_prompt(context, query):

    return f"""

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""
