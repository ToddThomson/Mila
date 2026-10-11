import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import mila
from mila_llm_server.model_worker import worker
from mila_llm_server.config import settings
from mila_llm_server.prompt import build_instruct_prompt

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = Field(settings.default_max_new_tokens, ge=1, le=4096)
    temperature: float = Field(settings.default_temperature, ge=0.0, le=2.0)
    top_k: int = Field(settings.default_top_k, ge=0)
    stream: bool = False
    system_prompt: str = Field("You are a helpful assistant.")


class ChatChoice(BaseModel):
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    object: str = "chat.completion"
    choices: list[ChatChoice]


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, http_req: Request):
    history = [m.model_dump() for m in req.messages[:-1]]
    user_message = req.messages[-1].content

    prompt_str = build_instruct_prompt(user_message, req.system_prompt, history)
    prompt_ids = await worker.encode(prompt_str)

    print(f"Prompt token count: {len(prompt_ids)}")
    print(f"First 10 tokens: {prompt_ids[:10]}")

    if req.stream:
        return StreamingResponse(
            _stream_chat(prompt_ids, req, http_req),
            media_type="text/event-stream",
        )

    output_ids = await worker.generate(
        prompt_ids, req.max_tokens, req.temperature, req.top_k
    )
    new_ids = output_ids[len(prompt_ids):]
    text = await worker.decode(new_ids)

    return ChatCompletionResponse(
        choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=text),
            finish_reason="stop",
        )]
    )


async def _stream_chat(
    prompt_ids: list[int],
    req: ChatCompletionRequest,
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
