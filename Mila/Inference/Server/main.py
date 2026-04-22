"""
Mila inference server — Llama 3.2 3B Instruct, CUDA BF16.
Start with: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os

cuda_path = os.environ.get("CUDA_PATH")
if cuda_path:
    os.add_dll_directory(os.path.join(cuda_path, "bin", "x64"))

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config import settings
from model_worker import worker
from routes.completions import router as completions_router
from routes.chat import router as chat_router
from routes.models import router as models_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.startup()
    yield
    await worker.shutdown()


app = FastAPI(
    title="Mila Inference Server",
    description="Llama 3.2 3B Instruct — CUDA BF16",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(completions_router)
app.include_router(chat_router)
app.include_router(models_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )