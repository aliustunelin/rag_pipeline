from fastapi import APIRouter, BackgroundTasks

from .service import IndexerService

router = APIRouter(prefix="/indexer", tags=["indexer"])

_service: IndexerService | None = None


def init_router(service: IndexerService):
    global _service
    _service = service


@router.post("/index")
async def trigger_indexing(background_tasks: BackgroundTasks):
    """Triggers full re-indexing of all data files."""
    background_tasks.add_task(_service.index_all)
    return {"message": "Indexing started in background"}


@router.post("/index/sync")
async def trigger_indexing_sync():
    """Triggers full re-indexing synchronously and returns result."""
    result = _service.index_all()
    return result


@router.get("/status")
async def get_status():
    """Returns current indexing status."""
    return _service.get_status()
