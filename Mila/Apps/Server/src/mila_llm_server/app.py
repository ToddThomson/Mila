"""
Mila inference server.
Start with: mila-server (or uvicorn mila_llm_server.app:app --host 0.0.0.0 --port 8000)
Protocol is selected via MILA_PROTOCOL env var: mila | openai | anthropic
"""
import logging
from contextlib import asynccontextmanager
from importlib.metadata import version

import mila
import uvicorn
from fastapi import FastAPI

from mila_llm_server.config import settings, ProtocolMode
from mila_llm_server.model_worker import worker
from mila_llm_server.routes.factory import register_routes
from mila_llm_server.routes.health import router as health_router
from mila_llm_server.protocols.openai import OpenAIAdapter
from mila_llm_server.protocols.anthropic import AnthropicAdapter
from mila_llm_server.protocols.mila import MilaChatAdapter

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
logging.basicConfig(
    level=logging.getLevelName(settings.log_level.upper()),
    format=_LOG_FORMAT,
    datefmt="%H:%M:%S",
    force=True,
)

_ADAPTERS = {
    ProtocolMode.openai: OpenAIAdapter,
    ProtocolMode.anthropic: AnthropicAdapter,
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
    # From the installed distribution rather than a literal, which was already two
    # release cycles stale. An src layout cannot be imported without being installed,
    # so the metadata is always there to read.
    version=version("mila-llm-server"),
    lifespan=lifespan,
)

register_routes(app, adapter)
app.include_router(health_router)


def main() -> None:
    """The mila-server console script. Serving is configuration, so it takes no arguments."""
    uvicorn.run(
        "mila_llm_server.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()