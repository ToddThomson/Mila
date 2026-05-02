"""
Mila inference server.
Start with: uvicorn main:app --host 0.0.0.0 --port 8000
Protocol is selected via MILA_PROTOCOL env var: mila | openai | anthropic
"""
import os

cuda_path = os.environ.get("CUDA_PATH")
if cuda_path:
    os.add_dll_directory(os.path.join(cuda_path, "bin", "x64"))

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config import settings, ProtocolMode
from model_worker import worker
from routes.factory import register_routes
from routes.models import router as models_router
from protocols.openai import OpenAIAdapter
from protocols.anthropic import AnthropicAdapter
from protocols.mila import MilaAdapter


_ADAPTERS = {
    ProtocolMode.openai: OpenAIAdapter,
    ProtocolMode.anthropic: AnthropicAdapter,
    ProtocolMode.mila: MilaAdapter,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.startup()
    yield
    await worker.shutdown()

adapter = _ADAPTERS[settings.protocol]()

app = FastAPI(
    title="Mila Inference Server",
    description=f"Protocol: {settings.protocol.value}",
    version="0.1.0",
    lifespan=lifespan,
)

register_routes(app, adapter)
app.include_router(models_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )