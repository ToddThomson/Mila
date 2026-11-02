"""
Protocol-agnostic health and root endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mila_llm_server.config import settings
from mila_llm_server.model_worker import worker

router = APIRouter()


def _status(**extra) -> JSONResponse:
    """`ok` while the model serves; `failed` with the reason once it has gone down, while the server shuts down."""
    if worker.failure is not None:
        return JSONResponse(status_code=503, content={"status": "failed", "error": worker.failure, **extra})

    return JSONResponse(content={"status": "ok", **extra})


@router.get("/", operation_id="health_root")
async def root() -> JSONResponse:
    return _status(protocol=settings.protocol.value)


@router.get("/v1/health", operation_id="health_check")
async def health() -> JSONResponse:
    return _status()
