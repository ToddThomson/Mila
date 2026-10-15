"""
Registers the active protocol's chat endpoints, completions and models routes. All generation, streaming and
reply parsing lives here exactly once, and knows nothing about which model is loaded: the prompt and the reply
are the model's chat protocol's (InferenceServerWire.md 3.3, 3.4).
"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

import mila
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mila_llm_server.config import loaded, settings
from mila_llm_server.conversation import RequestError, ServedCall, ServedReply, ServedRequest, call_id, placed_system_turns
from mila_llm_server.model_worker import worker
from mila_llm_server.protocols.base import ChatEndpoint, CompletionsCapable, ModelsCapable, ProtocolAdapter, ReplyContext
from mila_llm_server.protocols.utils import (
    END_TURN,
    STOP_SEQUENCE,
    TOOL_CALLS,
    StopScanner,
    finish_reason_from_status,
    truncate_at_stop,
)
from mila_llm_server.schemas.internal import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


def register_routes(app: FastAPI, adapter) -> None:
    if isinstance(adapter, ProtocolAdapter):
        for endpoint in adapter.chat_endpoints():
            _register_chat(app, endpoint)

    if isinstance(adapter, CompletionsCapable):
        _register_completions(app, adapter)

    if isinstance(adapter, ModelsCapable):
        _register_models(app, adapter)


def _register_chat(app: FastAPI, endpoint: ChatEndpoint) -> None:

    @app.post(endpoint.path, response_model=None)
    async def chat_endpoint(http_req: Request) -> JSONResponse | StreamingResponse:
        return await _serve_chat(endpoint, await http_req.json(), http_req)


def _register_completions(app: FastAPI, adapter: CompletionsCapable) -> None:

    @app.post(adapter.completions_path, response_model=None)
    async def completions_endpoint(http_req: Request) -> JSONResponse | StreamingResponse:
        body = await http_req.json()

        try:
            prompt_str, inf_req = adapter.parse_completions_request(body)
        except ValueError as error:
            return _invalid_request_error(str(error))

        return await _serve_completion(prompt_str, inf_req, http_req, adapter)


def _register_models(app: FastAPI, adapter: ModelsCapable) -> None:

    @app.get(adapter.models_path, response_model=None)
    async def list_models() -> JSONResponse:
        logger.info("GET %s called", adapter.models_path)
        return JSONResponse(content=adapter.format_models_response())


# ----------------------------------------------------------------------
# Chat: one conversation in, one parsed reply out
# ----------------------------------------------------------------------

async def _serve_chat(endpoint: ChatEndpoint, body: dict, http_req: Request) -> JSONResponse | StreamingResponse:
    protocol = worker.protocol

    if protocol is None:
        return JSONResponse(status_code=400, content=endpoint.error(
            f"'{loaded.name}' is a base model, which has no chat template; use the completions endpoint"))

    try:
        request = endpoint.parse(body, protocol)
        prompt = protocol.format_prompt(placed_system_turns(request.turns), request.template_variables, request.tools_json)
    except ValueError as error:
        # RequestError for what the endpoint refuses, ValueError for what the template refuses -- both name it.
        logger.warning("refused %s: %s; request shape: %s", endpoint.path, error, _shape(body))
        return JSONResponse(status_code=400, content=endpoint.error(str(error)))

    prompt_ids = await worker.encode(prompt["text"])
    remaining = loaded.context_length - len(prompt_ids)

    if remaining <= 0:
        return JSONResponse(status_code=400, content=endpoint.error(
            f"Prompt length {len(prompt_ids)} tokens exceeds context_length {loaded.context_length}."))

    requested = request.max_new_tokens or settings.default_max_new_tokens
    allowed = min(requested, remaining)

    context = ReplyContext(
        response_id=f"resp_{uuid.uuid4().hex}",
        model=loaded.name,
        created=int(time.time()),
        prompt_tokens=len(prompt_ids),
        wire=request.wire,
    )

    logger.info("prompt %d tokens; tail: %r", len(prompt_ids), prompt["text"][-600:])

    if request.stream:
        return StreamingResponse(
            _stream_chat(endpoint, request, prompt, prompt_ids, requested, allowed, context, http_req),
            media_type="text/event-stream")

    parts: list[str] = []
    status = await worker.generate_streaming(
        prompt_ids, parts.append, allowed, *_sampling(request), mila.StopController())

    reply = _reply(protocol, "".join(parts), prompt, status, request, requested, allowed, len(prompt_ids), len(parts))

    return JSONResponse(content=endpoint.format(reply, context))


def _shape(body: dict) -> str:
    """A refused request's structure without its text: the fields it set, and each message's role and part types."""
    messages = body.get("messages") or body.get("input") or []

    if isinstance(messages, str):
        messages = []

    def parts(message):
        content = message.get("content")

        if isinstance(content, list):
            return [part.get("type") for part in content if isinstance(part, dict)]

        return type(content).__name__

    settings_sent = {key: body[key] for key in sorted(body) if key not in ("messages", "input", "system", "instructions", "tools")}

    return f"settings {json.dumps(settings_sent)[:600]}; " + ", ".join(
        f"{message.get('role') or message.get('type')}:{parts(message)}" for message in messages if isinstance(message, dict))


def _sampling(request: ServedRequest) -> tuple[float, int, float]:
    return (
        settings.default_temperature if request.temperature is None else request.temperature,
        settings.default_top_k if request.top_k is None else request.top_k,
        settings.default_top_p if request.top_p is None else request.top_p,
    )


def _reply(protocol, raw: str, prompt: dict, status: str, request: ServedRequest, requested: int, allowed: int,
           prompt_tokens: int, completion_tokens: int) -> ServedReply:
    """
    The model's turn, read by its own protocol. A call the grammar could not read is never dispatched; it is
    answered as prose and logged (InferenceServerWire.md 3.2).
    """
    parsed = protocol.parse_reply(raw, prompt)
    content = parsed["content"]

    for span in parsed["malformed_tool_calls"]:
        logger.warning("the model wrote a tool call that does not parse; answering it as text: %r", span)
        content = f"{content}\n{span}".strip()

    calls = [ServedCall(call_id(call["name"], call["arguments"]), call["name"], call["arguments"])
             for call in parsed["tool_calls"]]

    if not request.parallel_tool_calls:
        calls = calls[:1]

    content, matched = truncate_at_stop(content, request.stop)

    if calls:
        finish_reason = TOOL_CALLS
    elif matched is not None:
        finish_reason = STOP_SEQUENCE
    else:
        finish_reason = finish_reason_from_status(status, requested, allowed)

    logger.info("reply: %d chars raw, %d calls, %d chars of reasoning", len(raw), len(calls), len(parsed["reasoning"]))

    return ServedReply(
        reasoning=parsed["reasoning"],
        content=content,
        tool_calls=calls,
        finish_reason=finish_reason,
        stop_sequence=matched,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _stream_chat(endpoint: ChatEndpoint, request: ServedRequest, prompt: dict, prompt_ids: list[int],
                       requested: int, allowed: int, context: ReplyContext, http_req: Request) -> AsyncIterator[str]:
    """
    A streamed reply, sent whole once the model's turn ends: what a span is -- reasoning, text or a call -- is
    known only once it is parsed. The stream opens before the prompt is read, and a keepalive goes out while
    the model works, so a client's read timeout does not fire (live streaming is Phase 4).
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop_ctrl = mila.StopController()
    loop = asyncio.get_running_loop()
    parts: list[str] = []

    def on_text(text: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, text)

    generation = asyncio.create_task(
        worker.generate_streaming(prompt_ids, on_text, allowed, *_sampling(request), stop_ctrl))
    generation.add_done_callback(lambda _: loop.call_soon_threadsafe(queue.put_nowait, None))

    yield endpoint.stream_open(context)

    last_text = time.monotonic()

    try:
        while True:
            if await http_req.is_disconnected():
                logger.info("client disconnected")
                stop_ctrl.request_stop()
                break

            try:
                text = await asyncio.wait_for(queue.get(), timeout=settings.keepalive_interval)
            except asyncio.TimeoutError:
                if parts and time.monotonic() - last_text > settings.decode_timeout:
                    logger.warning("decode_timeout hit after waiting %.1fs", settings.decode_timeout)
                    stop_ctrl.request_stop()
                    break

                yield endpoint.stream_keepalive(context)
                continue

            if text is None:
                break

            parts.append(text)
            last_text = time.monotonic()

    finally:
        if not generation.done():
            stop_ctrl.request_stop()
            await generation

        status = END_TURN if generation.cancelled() or generation.exception() is not None else generation.result()
        reply = _reply(worker.protocol, "".join(parts), prompt, status, request, requested, allowed,
                       len(prompt_ids), len(parts))

        yield endpoint.stream_reply(reply, context)


# ----------------------------------------------------------------------
# Completions: raw continuation, no conversation
# ----------------------------------------------------------------------

async def _serve_completion(prompt_str: str, inf_req: InferenceRequest, http_req: Request,
                            adapter: CompletionsCapable) -> JSONResponse | StreamingResponse:
    # A prompt may arrive as token ids, which are the model's input as sent: the caller tokenized it, and
    # encoding it again here would be a second tokenizer.
    prompt_ids = inf_req.prompt_ids or await worker.encode(prompt_str)
    remaining = loaded.context_length - len(prompt_ids)

    if remaining <= 0:
        return _prompt_too_long_error(len(prompt_ids), loaded.context_length)

    requested_tokens = inf_req.max_new_tokens
    inf_req.max_new_tokens = min(inf_req.max_new_tokens, remaining)
    inf_req.prompt_ids = prompt_ids

    if inf_req.stream:
        return StreamingResponse(_stream_completion(inf_req, http_req, adapter, requested_tokens), media_type="text/event-stream")

    output_ids, status = await worker.generate(
        inf_req.prompt_ids, inf_req.max_new_tokens, inf_req.temperature, inf_req.top_k, inf_req.top_p)
    new_ids = output_ids[len(inf_req.prompt_ids):]

    response = InferenceResponse(
        text=await worker.decode(new_ids),
        finish_reason=finish_reason_from_status(status, requested_tokens, inf_req.max_new_tokens),
        prompt_token_count=len(inf_req.prompt_ids),
        completion_token_count=len(new_ids),
    )

    # Cut after generation rather than during it: the caller receives the same reply, and stopping early would
    # save only the tokens past the stop.
    response.text, matched = truncate_at_stop(response.text, inf_req.stop)

    if matched is not None:
        response.finish_reason = STOP_SEQUENCE
        response.stop_sequence = matched

    return JSONResponse(content=adapter.format_completions_response(response))


async def _stream_completion(inf_req: InferenceRequest, http_req: Request, adapter: CompletionsCapable,
                             requested_tokens: int) -> AsyncIterator[str]:
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop_ctrl = mila.StopController()
    loop = asyncio.get_running_loop()
    # The same stop sequences a buffered reply is cut at, applied as the text arrives.
    scanner = StopScanner(inf_req.stop)

    def on_text(text: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, text)

    generation = asyncio.create_task(worker.generate_streaming(
        inf_req.prompt_ids, on_text, inf_req.max_new_tokens, inf_req.temperature, inf_req.top_k, inf_req.top_p,
        stop_ctrl, stop_at_markers=False))
    generation.add_done_callback(lambda _: loop.call_soon_threadsafe(queue.put_nowait, None))

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
            safe = scanner.feed(text)

            if safe:
                yield adapter.format_stream_chunk(safe, done=False)

            if scanner.matched is not None:
                stop_ctrl.request_stop()
                break

    finally:
        if not generation.done():
            stop_ctrl.request_stop()
            await generation

        held = scanner.flush()

        if held:
            yield adapter.format_stream_chunk(held, done=False)

        if scanner.matched is not None:
            finish_reason = STOP_SEQUENCE
        elif generation.cancelled() or generation.exception() is not None:
            finish_reason = END_TURN
        else:
            finish_reason = finish_reason_from_status(generation.result(), requested_tokens, inf_req.max_new_tokens)

        yield adapter.format_stream_chunk("", done=True, finish_reason=finish_reason)
        yield adapter.format_stream_done()


def _invalid_request_error(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
    )


def _prompt_too_long_error(prompt_length: int, context_length: int) -> JSONResponse:
    return _invalid_request_error(f"Prompt length {prompt_length} tokens exceeds context_length {context_length}.")
