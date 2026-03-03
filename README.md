# Rag Pipeline

A Retrieval-Augmented Generation system that reads company documents in different formats (TXT, CSV, JSON), finds the right information, and uses the most recent data when there is a conflict.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│ indexer_service  │     │   rag_service    │
│   (port 3334)   │     │   (port 1274)    │
│                 │     │                  │
│ Parse → Chunk   │────▶│ Search → Context │
│ Embed → FAISS   │     │ LLM → Stream     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
    ┌────▼───────────────────────▼────┐
    │     shared/faiss_store/         │
    │     (Docker Volume)             │
    └─────────────────────────────────┘
         │                       │
    ┌────▼───────────────────────▼────┐
    │           data/                 │
    │  sozlesme.txt                   │
    │  paket_fiyatlari.csv            │
    │  guncellemeler.json             │
    └─────────────────────────────────┘
```

### indexer_service
- Watches file changes with watchdog
- Uses a different parse/chunk strategy for each file type
- Creates embeddings with OpenAI text-embedding-3-small
- Stores vectors and metadata in FAISS

### rag_service
- Does semantic search in FAISS
- Gets the related part from the original file (using metadata)
- Generates a streaming answer with openai/gpt-4o-mini
- Temporal conflict resolution: always uses the most recent data

## Chunking Strategies

| File Type | Strategy | Why |
|---|---|---|
| **TXT** | Recursive Character Splitting (500 char, 100 overlap) | Good for long legal text |
| **CSV** | Row-based natural language conversion | Keeps the table structure |
| **JSON** | Each entry is a separate chunk + date metadata | Helps with temporal reasoning |

## Setup

### 1. Run Locally (python3)

Run both services in separate terminals. Start indexer_service first, then rag_service:

```bash
# Terminal 1 - Indexer Service
python3 indexer_service/app.py
```

```bash
# Terminal 2 - RAG Service
python3 rag_service/app.py
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

In Docker, `DATA_DIR` and `STORE_DIR` paths are overridden by the `environment` section in `docker-compose.yml`.

In the my local I changed docker and prod envs.

Services:
- **indexer_service**: http://localhost:3334
- **rag_service**: http://localhost:1274

On the first run, indexer_service automatically indexes all files in the data/ folder.

## API Endpoints

### RAG Service (port 1274)

#### POST /rag/query
Full RAG query - returns an answer with sources.

```bash
curl -X POST http://localhost:1274/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Pro package price and when do I get my refund if I cancel?", "top_k": 5}'
```

#### POST /rag/query/stream
Streaming RAG query - returns the answer as an SSE stream.

```bash
curl -X POST http://localhost:1274/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Basic package storage limit?"}'
```

#### POST /rag/reload
Reloads the FAISS index from disk.

```bash
curl -X POST http://localhost:1274/rag/reload
```

### Indexer Service (port 3334)

#### POST /indexer/index
Starts re-indexing in the background.

#### POST /indexer/index/sync
Runs indexing and waits for the result.

#### GET /indexer/status
Returns the current index status.

## Testing Dynamic Updates

The system detects file changes automatically. To test:

1. Change the Pro price in the CSV:
   ```
   Pro,399,3999,10000,50,7/24 Live Support,14,5
   ```

2. Add a new update to the JSON:
   ```json
   {"tarih": "2025-01-15", "etkilenen_paket": "Basic", "degisiklik": "Basic package refund period reduced to 7 days.", "onceki_deger": "14 days", "yeni_deger": "7 days"}
   ```

3. Send a query and check that the answer uses the updated data.

## Tech Stack

- **Framework**: FastAPI (async, streaming)
- **Embedding**: OpenAI text-embedding-3-small
- **Vector DB**: FAISS (file-based, lightweight)
- **LLM**: OpenAI gpt-4o-mini
- **File Watcher**: watchdog
- **Container**: Docker + Docker Compose

## Future Improvements

- **Vector DB**: FAISS can be replaced with Qdranti, Milvus vs. It supports different metrics like cosine similarity and dot product for better retrieval.
- **Index Update Strategy**: Right now, watchdog detects file changes and re-indexes all files from scratch. Indexing only the changed parts would be cheaper.
- **GPU-Powered Search**: Milvus or Qdrant GPU-backed modes can improve search speed for large datasets.
- **Self-Hosted Models**: Currently using API-based models for demo purposes. In production, we can serve our own models with vLLM or SGLang to reduce cost and latency.
- **Service Communication**: If services need to talk to each other, gRPC can be used instead of REST for faster binary communication. This would be for internal services only — external clients would still use REST or WebSocket.