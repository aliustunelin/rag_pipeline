from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..service.main_service import MainService

router = APIRouter(prefix="/rag", tags=["rag"])

_main_service: MainService | None = None


def init_router(main_service: MainService):
    global _main_service
    _main_service = main_service


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/query")
async def query(request: QueryRequest):
    """Full RAG query - returns complete answer with sources."""
    _main_service.top_k = request.top_k
    result = await _main_service.query(request.query)
    return result


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming RAG query - returns answer as SSE stream."""
    _main_service.top_k = request.top_k

    async def event_generator():
        async for chunk in _main_service.query_stream(request.query):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/reload")
async def reload_index():
    """Reloads FAISS index from disk."""
    success = _main_service.reload_index()
    if success:
        return {"status": "Index reloaded successfully"}
    return {"status": "Failed to reload index", "detail": "Index files not found"}
