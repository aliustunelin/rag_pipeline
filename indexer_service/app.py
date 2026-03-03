import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI

env_prod = Path(__file__).resolve().parent.parent / ".env.prod"
if env_prod.exists():
    load_dotenv(env_prod, override=False)

from src.router import router, init_router
from src.service import IndexerService
from src.watcher import DataWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR")
STORE_DIR = os.getenv("STORE_DIR")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

watcher: DataWatcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global watcher

    service = IndexerService(
        data_dir=DATA_DIR,
        store_dir=STORE_DIR,
        OPENROUTER_API_KEY=OPENROUTER_API_KEY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL,
    )
    init_router(service)

    logger.info("Running initial indexing...")
    service.index_all()

    watcher = DataWatcher(data_dir=DATA_DIR, on_change_callback=service.index_all)
    watcher.start()

    yield

    if watcher:
        watcher.stop()


app = FastAPI(
    title="Indexer Service",
    description="Document indexing pipeline: parse → chunk → embed → FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "indexer"}


if __name__ == "__main__":
    host = os.getenv("INDEXER_HOST")
    port = int(os.getenv("INDEXER_PORT"))
    uvicorn.run("app:app", host=host, port=port)
