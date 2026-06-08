"""
Registers chat, completions, and (when the adapter supports it) responses
and models routes for whichever protocol adapter is active. All
streaming/queueing mechanics live here exactly once.
"""
import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

import mila
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_worker import worker
from protocols.base import ProtocolAdapter, ResponsesCapable, ModelsCapable
from schemas.internal import InferenceRequest, InferenceResponse
from config import settings

logger = logging.getLogger(__name__)


def register_routes(app: FastAPI, adapter) -> None:
    if isinstance(adapter, ProtocolAdapter):
        _register_chat(app, adapter)
        _register_completions(app, adapter)

    if isinstance(adapter, ResponsesCapable):
        _register_responses(app, adapter)

    if isinstance(adapter, ModelsCapable):
        _register_models(app, adapter)


def _register_chat(app: FastAPI, adapter: ProtocolAdapter) -> None:

    @app.post(adapter.chat_path, response_model=None)
    async def chat_endpoint(http_req: Request) -> JSONResponse | StreamingResponse:
        body = await http_req.json()
        prompt_str, inf_req = adapter.parse_chat_request(body)
        return await _dispatch(prompt_str, inf_req, http_req, adapter, is_chat=True)


def _register_completions(app: FastAPI, adapter: ProtocolAdapter) -> None:

    @app.post(adapter.completions_path, response_model=None)
    async def completions_endpoint(http_req: Request) -> JSONResponse | StreamingResponse:
        body = await http_req.json()
        prompt_str, inf_req = adapter.parse_completions_request(body)
        return await _dispatch(prompt_str, inf_req, http_req, adapter, is_chat=False)


def _register_responses(app: FastAPI, adapter: ResponsesCapable) -> None:

    @app.post(adapter.responses_path, response_model=None)
    async def responses_endpoint(http_req: Request) -> JSONResponse | StreamingResponse:
        body = await http_req.json()
        prompt_str, inf_req = adapter.parse_responses_request(body)
        return await _dispatch_responses(prompt_str, inf_req, http_req, adapter)


def _register_models(app: FastAPI, adapter: ModelsCapable) -> None:

    @app.get(adapter.models_path, response_model=None)
    async def list_models() -> JSONResponse:
        logger.info("GET %s called", adapter.models_path)
        return JSONResponse(content=adapter.format_models_response())


async def _dispatch(
    prompt_str: str,
    inf_req: InferenceRequest,
    http_req: Request,
    adapter: ProtocolAdapter,
    is_chat: bool,
) -> JSONResponse | StreamingResponse:
    prompt_ids = await worker.encode(prompt_str)

    remaining = settings.context_length - len(prompt_ids)
    if remaining <= 0:
        return _prompt_too_long_error(len(prompt_ids), settings.context_length)

    inf_req.max_new_tokens = min(inf_req.max_new_tokens, remaining)
    inf_req.prompt_ids = prompt_ids

    if inf_req.stream:
        return StreamingResponse(
            _stream(inf_req, http_req, adapter),
            media_type="text/event-stream",
        )

    output_ids = await worker.generate(
        inf_req.prompt_ids,
        inf_req.max_new_tokens,
        inf_req.temperature,
        inf_req.top_k,
        inf_req.top_p,
    )

    new_ids = output_ids[len(inf_req.prompt_ids):]
    text = await worker.decode(new_ids)

    response = InferenceResponse(
        text=text,
        finish_reason="stop",
        prompt_token_count=len(inf_req.prompt_ids),
        completion_token_count=len(new_ids),
    )
    payload = adapter.format_chat_response(response) if is_chat else adapter.format_completions_response(response)
    return JSONResponse(content=payload)


async def _dispatch_responses(
    prompt_str: str,
    inf_req: InferenceRequest,
    http_req: Request,
    adapter: ResponsesCapable,
) -> JSONResponse | StreamingResponse:
    prompt_ids = await worker.encode(prompt_str)

    remaining = settings.context_length - len(prompt_ids)
    if remaining <= 0:
        return _prompt_too_long_error(len(prompt_ids), settings.context_length)

    inf_req.max_new_tokens = min(inf_req.max_new_tokens, remaining)
    inf_req.prompt_ids = prompt_ids

    response_id = f"resp-{uuid.uuid4().hex}"

    if inf_req.stream:
        return StreamingResponse(
            _stream_responses(inf_req, http_req, adapter, response_id),
            media_type="text/event-stream",
        )

    output_ids = await worker.generate(
        inf_req.prompt_ids,
        inf_req.max_new_tokens,
        inf_req.temperature,
        inf_req.top_k,
        inf_req.top_p,
    )

    new_ids = output_ids[len(inf_req.prompt_ids):]
    text = await worker.decode(new_ids)

    response = InferenceResponse(
        text=text,
        finish_reason="stop",
        prompt_token_count=len(inf_req.prompt_ids),
        completion_token_count=len(new_ids),
    )
    return JSONResponse(content=adapter.format_responses_response(response))


async def _stream(
    inf_req: InferenceRequest,
    http_req: Request,
    adapter: ProtocolAdapter,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop_ctrl = mila.StopController()
    loop = asyncio.get_running_loop()

    def on_text(text: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, text)

    def on_done() -> None:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    generation = asyncio.create_task(
        worker.generate_streaming(
            inf_req.prompt_ids,
            on_text,
            inf_req.max_new_tokens,
            inf_req.temperature,
            inf_req.top_k,
            inf_req.top_p,
            stop_ctrl,
        )
    )
    generation.add_done_callback(lambda _: on_done())

    if hasattr(adapter, "format_stream_preamble"):
        yield adapter.format_stream_preamble(len(inf_req.prompt_ids))

    output_token_count = 0
    first_token = True

    try:
        while True:
            if await http_req.is_disconnected():
                logger.info("client disconnected")
                stop_ctrl.request_stop()
                break

            timeout = settings.keepalive_interval if first_token else settings.decode_timeout

            try:
                text = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if first_token:
                    yield ": keepalive\n\n"
                    continue
                logger.warning("decode_timeout hit after waiting %.1fs", settings.decode_timeout)
                stop_ctrl.request_stop()
                break

            if text is None:
                break

            first_token = False
            output_token_count += 1
            yield adapter.format_stream_chunk(text, done=False)

    finally:
        if not generation.done():
            stop_ctrl.request_stop()
            await generation
        yield adapter.format_stream_chunk("", done=True)
        if hasattr(adapter, "format_stream_message_delta"):
            yield adapter.format_stream_message_delta(output_token_count)
        yield adapter.format_stream_done()


async def _stream_responses(
    inf_req: InferenceRequest,
    http_req: Request,
    adapter: ResponsesCapable,
    response_id: str,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop_ctrl = mila.StopController()
    loop = asyncio.get_running_loop()
    accumulated: list[str] = []
    t_start = time.time()

    def _elapsed() -> str:
        return f"{time.time() - t_start:.2f}s"

    def on_text(text: str) -> None:
        accumulated.append(text)
        loop.call_soon_threadsafe(queue.put_nowait, text)

    def on_done() -> None:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    generation = asyncio.create_task(
        worker.generate_streaming(
            inf_req.prompt_ids,
            on_text,
            inf_req.max_new_tokens,
            inf_req.temperature,
            inf_req.top_k,
            inf_req.top_p,
            stop_ctrl,
        )
    )
    generation.add_done_callback(lambda _: on_done())

    print(f"[{_elapsed()}] {response_id}: yielding response.created", flush=True)
    yield adapter.format_responses_stream_created(response_id)

    first_token = True
    token_count = 0

    try:
        while True:
            disconnected = await http_req.is_disconnected()
            if disconnected:
                print(f"[{_elapsed()}] {response_id}: client disconnected after {token_count} tokens", flush=True)
                stop_ctrl.request_stop()
                break

            timeout = settings.keepalive_interval if first_token else settings.decode_timeout

            try:
                text = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if first_token:
                    print(f"[{_elapsed()}] {response_id}: keepalive ping", flush=True)
                    yield adapter.format_responses_stream_keepalive(response_id)
                    continue
                print(f"[{_elapsed()}] {response_id}: decode_timeout after {token_count} tokens", flush=True)
                stop_ctrl.request_stop()
                break

            if text is None:
                print(f"[{_elapsed()}] {response_id}: generation done after {token_count} tokens", flush=True)
                break

            if first_token:
                print(f"[{_elapsed()}] {response_id}: first token received", flush=True)
                first_token = False

            token_count += 1

    finally:
        if not generation.done():
            stop_ctrl.request_stop()
            await generation

        full_text = "".join(accumulated)
        print(f"[{_elapsed()}] {response_id}: full_text preview: {repr(full_text[:200])}", flush=True)

        tool_call_item = adapter.parse_tool_call_from_text(full_text) if hasattr(adapter, "parse_tool_call_from_text") else None

        if tool_call_item:
            print(f"[{_elapsed()}] {response_id}: tool call detected — {tool_call_item['name']}", flush=True)
            yield adapter.format_responses_stream_function_call(response_id, tool_call_item)
            yield adapter.format_responses_stream_done_with_tool_call(response_id, tool_call_item)
        else:
            item_id = f"msg-{uuid.uuid4().hex}"
            yield adapter.format_responses_stream_output_item_added(response_id, item_id)
            yield adapter.format_responses_stream_content_part_added(response_id)
            yield adapter.format_responses_stream_chunk(full_text, done=True, response_id=response_id)
            yield adapter.format_responses_stream_content_part_done(response_id, full_text)
            yield adapter.format_responses_stream_output_item_done(response_id, item_id, full_text)
            yield adapter.format_responses_stream_done(response_id, output_text=full_text)

        print(f"[{_elapsed()}] {response_id}: stream closed", flush=True)


def _prompt_too_long_error(prompt_length: int, context_length: int) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": f"Prompt length {prompt_length} tokens exceeds context_length {context_length}.",
            },
        },
    )