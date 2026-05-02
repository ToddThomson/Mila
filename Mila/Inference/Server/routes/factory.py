"""
Registers chat and completions routes for whichever protocol adapter is active.
All streaming/queueing mechanics live here exactly once.
"""
import asyncio
from typing import AsyncIterator

import mila
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_worker import worker
from protocols.base import ProtocolAdapter
from schemas.internal import InferenceRequest, InferenceResponse
from config import settings


def register_routes(app: FastAPI, adapter: ProtocolAdapter) -> None:
    _register_chat(app, adapter)
    _register_completions(app, adapter)


def _register_chat(app: FastAPI, adapter: ProtocolAdapter) -> None:

    @app.post(adapter.chat_path, response_model=None)
    async def chat_endpoint(http_req: Request) -> JSONResponse | StreamingResponse:
        body = await http_req.json()
        prompt_str, inf_req = adapter.parse_chat_request(body)
        return await _dispatch(prompt_str, inf_req, http_req, adapter, is_chat=True)


def _register_completions(app: FastAPI, adapter: ProtocolAdapter) -> None:

    @app.post(adapter.completions_path, response_model=None )
    async def completions_endpoint(http_req: Request) -> JSONResponse | StreamingResponse:
        body = await http_req.json()
        prompt_str, inf_req = adapter.parse_completions_request(body)
        return await _dispatch(prompt_str, inf_req, http_req, adapter, is_chat=False)


async def _dispatch(
    prompt_str: str,
    inf_req: InferenceRequest,
    http_req: Request,
    adapter: ProtocolAdapter,
    is_chat: bool,
) -> JSONResponse | StreamingResponse:
    prompt_ids = await worker.encode(prompt_str)
    print(f"[dispatch] encoded, token count={len(prompt_ids)}, max_new={inf_req.max_new_tokens}")

    # Clamp max_new_tokens to the remaining context budget.
    remaining = settings.context_length - len(prompt_ids)
    if remaining <= 0:
        return JSONResponse(
            status_code=400,
            content={"error": f"Prompt length {len(prompt_ids)} exceeds context_length {settings.context_length}"}
        )
    inf_req.max_new_tokens = min(inf_req.max_new_tokens, remaining)
    print(f"[dispatch] clamped max_new_tokens to {inf_req.max_new_tokens}")

    inf_req.prompt_ids = prompt_ids

    if inf_req.stream:
        print(f"[dispatch] streaming mode")
        stream = _stream(inf_req, http_req, adapter)
        return StreamingResponse(stream, media_type="text/event-stream")

    print(f"[dispatch] starting generate")
    output_ids = await worker.generate(
        inf_req.prompt_ids,
        inf_req.max_new_tokens,
        inf_req.temperature,
        inf_req.top_k,
    )
    print(f"[dispatch] generate complete, output tokens={len(output_ids)}")

    new_ids = output_ids[len(inf_req.prompt_ids):]
    text = await worker.decode(new_ids)
    print(f"[dispatch] decoded: {text[:80]!r}")

    response = InferenceResponse(
        text=text,
        finish_reason="stop",
        prompt_token_count=len(inf_req.prompt_ids),
        completion_token_count=len(new_ids),
    )

    if is_chat:
        payload = adapter.format_chat_response(response)
    else:
        payload = adapter.format_completions_response(response)

    return JSONResponse(content=payload)


async def _stream(
    inf_req: InferenceRequest,
    http_req: Request,
    adapter: ProtocolAdapter,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    stop_ctrl = mila.StopController()
    loop = asyncio.get_running_loop()

    def on_token(token_id: int) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, token_id)

    def on_done() -> None:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    generation = asyncio.create_task(
        worker.generate_streaming(
            inf_req.prompt_ids,
            on_token,
            inf_req.max_new_tokens,
            inf_req.temperature,
            inf_req.top_k,
            stop_ctrl,
        )
    )
    generation.add_done_callback(lambda _: on_done())

    try:
        while True:
            if await http_req.is_disconnected():
                print("[stream] client disconnected")
                stop_ctrl.request_stop()
                break

            try:
                token_id = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                print("[stream] timeout waiting for token")
                stop_ctrl.request_stop()
                break

            if token_id is None:
                print("[stream] done sentinel received")
                break

            text = await worker.decode([token_id])
            print(f"[stream] token: {text!r}")
            yield adapter.format_stream_chunk(text, done=False)

    finally:
        if not generation.done():
            stop_ctrl.request_stop()
            await generation
        yield adapter.format_stream_chunk("", done=True)
        yield adapter.format_stream_done()
