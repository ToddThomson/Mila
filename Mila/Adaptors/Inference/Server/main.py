"""
Mila inference server.
Start with: uvicorn main:app --host 0.0.0.0 --port 8000
Protocol is selected via MILA_PROTOCOL env var: mila | openai | anthropic
"""
import logging
from contextlib import asynccontextmanager

import mila
import uvicorn
from fastapi import FastAPI

from config import settings, ProtocolMode
from model_worker import worker
from routes.factory import register_routes
from routes.health import router as health_router
from protocols.openai import OpenAIAdapter
from protocols.anthropic import AnthropicMessagesAdapter
from protocols.mila import MilaChatAdapter

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
logging.basicConfig(
    level=logging.getLevelName(settings.log_level.upper()),
    format=_LOG_FORMAT,
    datefmt="%H:%M:%S",
    force=True,
)

_ADAPTERS = {
    ProtocolMode.openai: OpenAIAdapter,
    ProtocolMode.anthropic: AnthropicMessagesAdapter,
    ProtocolMode.mila: MilaChatAdapter,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mila.initialize(log_level=settings.log_level.lower())
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
app.include_router(health_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )