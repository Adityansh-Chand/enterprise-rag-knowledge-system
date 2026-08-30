FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

# CPU torch explicitly: the default resolution pulls the CUDA build (~2GB) for a
# workload that never uses a GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# The service defaults to RETRIEVER=bm25, which needs no model download and
# starts instantly. Set RETRIEVER=hybrid to use dense retrieval; the embedding
# model is then fetched on first use unless it is baked in or mounted.
ENV RETRIEVER=bm25

CMD ["uvicorn","api.server:app","--host","0.0.0.0","--port","8000"]
