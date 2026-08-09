from fastapi import APIRouter
from pydantic import BaseModel

from model_worker import worker
from config import settings

router = APIRouter()


class ModelInfo(BaseModel):
    id: str = settings.model
    object: str = "model"
    config: dict


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


@router.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "protocol": settings.protocol.value}

@router.get("/v1/models", response_model=ModelList)
async def list_models():
    config = await worker.get_model_info()
    return ModelList(data=[ModelInfo(config=config)])


@router.get("/v1/health")
async def health():
    return {"status": "ok"}
