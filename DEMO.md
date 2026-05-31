# Demo

This demo shows the RAG service answering an HR policy question, exposing metrics,
and recording an audit event.

## Run Locally

Terminal 1:

```bash
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000
```

Terminal 2:

```bash
python scripts/smoke_test.py
```

To demo protected endpoints, start with an API key:

```bash
API_KEY=demo-key uvicorn api.server:app --reload --port 8000
```

## Curl Walkthrough

Health:

```bash
curl http://localhost:8000/health
```

Metrics:

```bash
curl http://localhost:8000/metrics
```

GET query:

```bash
curl "http://localhost:8000/query?q=remote%20work%20policy"
```

POST query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d @examples/requests/query.json
```

Events when `API_KEY` is set:

```bash
curl http://localhost:8000/events \
  -H "X-API-Key: demo-key"
```

Protected POST query when `API_KEY` is set:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key" \
  -d @examples/requests/query.json
```

## Sample Files

- Request: `examples/requests/query.json`
- Responses: `examples/responses/health.json`, `metrics.json`, `query.json`, `events.json`
