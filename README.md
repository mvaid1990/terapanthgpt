# Religious RAG Stack (n8n approvals + Weaviate + Open-WebUI)

## Services
- Open-WebUI: http://localhost:3000
- n8n: http://localhost:5678
- Weaviate: http://localhost:8080
- MinIO API: http://localhost:9000
- MinIO Console: http://localhost:9001
- Ingestion API (admin + OpenAI-compatible): http://localhost:8000

## Credentials (dev)
- Ingestion API key: `devkey`
- MinIO:
  - user: `minio`
  - password: `minio12345`
- Postgres (n8n DB):
  - user: `n8n`
  - password: `n8n`
  - db: `n8n`

## Artifact storage
By default, the stack is configured to store originals/extracted text on the ingestion container filesystem:
- `STORAGE_BACKEND=local`
- `ARTIFACTS_DIR=/data/artifacts`

These artifacts live in the `ingestion_data` docker volume.

If you want to switch back to MinIO later:
- Set `STORAGE_BACKEND=minio`
- Set `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, and `MINIO_BUCKET` in `docker-compose.yml`
- Add `minio` back into `ingestion_api.depends_on`

## Start
Run docker compose from this directory.

## Pull local models (Ollama)
The stack uses:
- chat model: `llama3.1:8b`
- embedding model: `nomic-embed-text`

After the stack is up, pull models:
- `ollama pull llama3.1:8b`
- `ollama pull nomic-embed-text`

## Smoke test (no n8n yet)
### 1) Create a source (text)
```bash
curl -s \
  -H 'Authorization: Bearer devkey' \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"text","title":"Test Doc","text":"This is a test document about compassion and charity.","tags":["test"]}' \
  http://localhost:8000/admin/sources | jq
```

### 1b) Create a source (YouTube)
```bash
curl -s \
  -H 'Authorization: Bearer devkey' \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"youtube","title":"YT Test","source_uri":"https://www.youtube.com/watch?v=VIDEO_ID","tags":["test"]}' \
  http://localhost:8000/admin/sources | jq
```

Notes:
- Captions-first; if disabled/unavailable, it falls back to Whisper.
- Whisper runs on CPU by default and may be slow on a small VM.

### 1c) Create a source (PDF)
```bash
curl -s \
  -H 'Authorization: Bearer devkey' \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"pdf","title":"PDF Test","source_uri":"https://example.com/file.pdf","tags":["test"]}' \
  http://localhost:8000/admin/sources | jq
```

Notes:
- For scanned PDFs, OCR is attempted when extracted text is too short.
- You can control OCR via `OCR_ENABLED` and `OCR_MAX_PAGES` in compose.

### 1d) Create a source (EPUB)
```bash
curl -s \
  -H 'Authorization: Bearer devkey' \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"epub","title":"EPUB Test","source_uri":"https://example.com/book.epub","tags":["test"]}' \
  http://localhost:8000/admin/sources | jq
```

Copy the returned `id`.

### 2) Approve
```bash
curl -s -X POST \
  -H 'Authorization: Bearer devkey' \
  http://localhost:8000/admin/sources/<ID>/approve | jq
```

### 3) Index into Weaviate
```bash
curl -s -X POST \
  -H 'Authorization: Bearer devkey' \
  http://localhost:8000/admin/sources/<ID>/index | jq
```

### 4) Chat via OpenAI-compatible endpoint
```bash
curl -s \
  -H 'Authorization: Bearer devkey' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What does the test doc say about charity?"}]}' \
  http://localhost:8000/v1/chat/completions | jq -r '.choices[0].message.content'
```

## n8n approval flow (manual build)
You can build two simple workflows in n8n:

### Workflow A: Submit Source
- Trigger: Webhook (POST) `/submit-source`
- Node: HTTP Request -> POST `http://ingestion_api:8000/admin/sources`
  - Headers:
    - `Authorization: Bearer devkey`
    - `Content-Type: application/json`
  - Body: JSON from webhook (at minimum `source_type=text`, `text`, optional `title`, `tags`)
- Node: Respond to Webhook -> return the ingestion API response (contains `id` and `status=awaiting_approval`)

### Workflow B: Approve + Index
- Trigger: Webhook (POST) `/approve/{source_id}`
- Node 1: HTTP Request -> POST `http://ingestion_api:8000/admin/sources/{{$json.params.source_id}}/approve`
- Node 2: HTTP Request -> POST `http://ingestion_api:8000/admin/sources/{{$json.params.source_id}}/index`
- Respond to Webhook -> return final status

You can add a third workflow for reject:
- Webhook (POST) `/reject/{source_id}` -> POST `/admin/sources/{id}/reject`

## Notes / current limitations
- Supported source types:
  - `text`
  - `youtube` (captions-first, Whisper fallback)
  - `pdf` (text extraction + optional OCR)
  - `epub`
- Source content is stored in MinIO under `sources/<source_id>/` (e.g. `original.pdf`, `original.epub`, `extracted.txt`, `metadata.json`).
