from fastapi import APIRouter
from pydantic import BaseModel

from model_worker import worker

router = APIRouter()


class ModelInfo(BaseModel):
    id: str = "llama-3.2-3b-instruct"
    object: str = "model"
    config: dict


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


@router.get("/v1/models", response_model=ModelList)
async def list_models():
    config = await worker.get_model_info()
    return ModelList(data=[ModelInfo(config=config)])


@router.get("/health")
async def health():
    return {"status": "ok"}
