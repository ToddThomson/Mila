"""
Protocol-agnostic health and root endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mila_llm_server.config import settings

router = APIRouter()


@router.get("/", operation_id="health_root")
async def root() -> JSONResponse:
    return JSONResponse(content={"status": "ok", "protocol": settings.protocol.value})


@router.get("/v1/health", operation_id="health_check")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})