"""
Mila inference server.
Start with: mila-server (or uvicorn mila_llm_server.app:app --host 0.0.0.0 --port 8000)
Protocol is selected via MILA_PROTOCOL env var: openai | anthropic
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


def serve() -> None:
    """
    Run the server, as mila-server does when given no command. Serving is configuration, so it takes none. A model
    that fails takes the server down: it finishes the replies in flight, the failing one with its error, and exits 1
    so whatever started it can tell a failure from a stop.
    """
    server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level=settings.log_level))
    # Set from the worker thread; uvicorn reads the flag on its own loop, so no hand-off is needed.
    worker.on_failure = lambda: setattr(server, "should_exit", True)
    server.run()

    if worker.failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    serve()