from fastapi import APIRouter
from pydantic import BaseModel

from model_worker import worker
from config import settings, loaded

router = APIRouter()


class ModelInfo(BaseModel):
    id: str = settings.model
    object: str = "model"
    config: dict

    # Beyond OpenAI's model object, which carries no lineage. A client that lists models is
    # the one place a served model is presented to a person, so a license requiring displayed
    # attribution is answered here; `attribution` is empty when the license asks for none.
    base_model: str = ""
    license: str = ""
    attribution: str = ""


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


@router.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "protocol": settings.protocol.value}

@router.get("/v1/models", response_model=ModelList)
async def list_models():
    config = await worker.get_model_info()

    return ModelList(data=[ModelInfo(
        config=config,
        base_model=loaded.base_model,
        license=loaded.license,
        attribution=loaded.attribution,
    )])


@router.get("/v1/health")
async def health():
    return {"status": "ok"}
