import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import mila
from model_worker import worker
from config import settings

router = APIRouter()


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(settings.default_max_new_tokens, ge=1, le=4096)
    temperature: float = Field(settings.default_temperature, ge=0.0, le=2.0)
    top_k: int = Field(settings.default_top_k, ge=0)
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    object: str = "text_completion"
    choices: list[CompletionChoice]


@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest, http_req: Request):
    prompt_ids = await worker.encode(req.prompt)

    if req.stream:
        return StreamingResponse(
            _stream_completion(prompt_ids, req, http_req),
            media_type="text/event-stream",
        )

    output_ids = await worker.generate(
        prompt_ids, req.max_tokens, req.temperature, req.top_k
    )
    new_ids = output_ids[len(prompt_ids):]
    text = await worker.decode(new_ids)

    return CompletionResponse(choices=[CompletionChoice(text=text, finish_reason="stop")])


async def _stream_completion(
    prompt_ids: list[int],
    req: CompletionRequest,
    http_req: Request,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    stop_ctrl = mila.StopController()
    loop = asyncio.get_running_loop()

    def on_token(token_id: int) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, token_id)

    generation = asyncio.create_task(
        worker.generate_streaming(
            prompt_ids, on_token, req.max_tokens, req.temperature, req.top_k, stop_ctrl
        )
    )

    try:
        while True:
            if await http_req.is_disconnected():
                stop_ctrl.request_stop()
                break

            try:
                token_id = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                stop_ctrl.request_stop()
                break

            if token_id is None:
                break

            text = await worker.decode([token_id])
            yield f"data: {text}\n\n"

    finally:
        if not generation.done():
            stop_ctrl.request_stop()
            await generation
        yield "data: [DONE]\n\n"
