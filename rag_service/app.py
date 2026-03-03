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

from src.router.query_router import router, init_router
from src.service.retriever import Retriever
from src.service.context_builder import ContextBuilder
from src.service.llm_client import LLMClient
from src.service.main_service import MainService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STORE_DIR = os.getenv("STORE_DIR")
DATA_DIR = os.getenv("DATA_DIR")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
TOP_K = int(os.getenv("TOP_K"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = Retriever(
        store_dir=STORE_DIR,
        OPENROUTER_API_KEY=OPENROUTER_API_KEY,
        embedding_model=EMBEDDING_MODEL,
    )
    retriever.load_index()

    context_builder = ContextBuilder(data_dir=DATA_DIR)

    llm_client = LLMClient(api_key=OPENROUTER_API_KEY, model=LLM_MODEL)

    main_service = MainService(
        retriever=retriever,
        context_builder=context_builder,
        llm_client=llm_client,
        top_k=TOP_K,
    )
    init_router(main_service)

    logger.info("RAG service initialized")
    yield


app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation: search → context → LLM → streaming response",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rag"}


if __name__ == "__main__":
    host = os.getenv("RAG_HOST")
    port = int(os.getenv("RAG_PORT"))
    uvicorn.run("app:app", host=host, port=port)
