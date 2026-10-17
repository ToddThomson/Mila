"""
Mila-native protocol adapter -- minimal, no ceremony.
  Chat:        POST /v1/chat/completions
  Completions: POST /v1/completions
"""
import json
import time
import uuid

from mila_llm_server.conversation import (
    RequestError, ServedReply, ServedRequest, assistant_turn, presence_penalty_of, text_of)
from mila_llm_server.protocols.base import ChatEndpoint, CompletionsCapable, ProtocolAdapter, ReplyContext
from mila_llm_server.protocols.utils import END_TURN, openai_finish_reason
from mila_llm_server.schemas.internal import InferenceRequest, InferenceResponse


class MilaChat(ChatEndpoint):

    @property
    def path(self) -> str:
        return "/v1/chat/completions"

    def parse(self, body: dict, protocol) -> ServedRequest:
        turns = []

        if body.get("system_prompt"):
            turns.append({"role": "system", "content": body["system_prompt"]})

        for index, message in enumerate(body.get("messages") or []):
            role = message.get("role")
            content = text_of(message.get("content"), field_name=f"messages[{index}].content")

            if role in ("system", "user"):
                turns.append({"role": role, "content": content})
            elif role == "assistant":
                turns.append(assistant_turn(content))
            else:
                raise RequestError(f"messages[{index}].role '{role}' is not one this server reads")

        return ServedRequest(
            turns=turns,
            max_new_tokens=body.get("max_new_tokens", 256),
            temperature=body.get("temperature"),
            top_k=body.get("top_k"),
            top_p=body.get("top_p"),
            presence_penalty=presence_penalty_of(body),
            stream=bool(body.get("stream", False)),
        )

    def format(self, reply: ServedReply, context: ReplyContext) -> dict:
        return {
            "id": context.response_id,
            "object": "chat.completion",
            "created": context.created,
            "model": "mila",
            "text": reply.content,
            "reasoning": reply.reasoning,
            "finish_reason": openai_finish_reason(reply.finish_reason),
            "usage": {"prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens},
        }

    def stream_open(self, context: ReplyContext) -> str:
        return ""

    def stream_keepalive(self, context: ReplyContext) -> str:
        return ": keepalive\n\n"

    def stream_reply(self, reply: ServedReply, context: ReplyContext) -> str:
        chunk = {"text": reply.content, "done": True, "finish_reason": openai_finish_reason(reply.finish_reason)}

        return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"


class MilaCompletions(CompletionsCapable):

    @property
    def completions_path(self) -> str:
        return "/v1/completions"

    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]:
        prompt_str = body.get("prompt", "")
        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_new_tokens", 256),
            temperature=body.get("temperature"),
            top_k=body.get("top_k"),
            top_p=body.get("top_p"),
            presence_penalty=presence_penalty_of(body),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def format_completions_response(self, response: InferenceResponse) -> dict:
        return {
            "id": f"mila-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "mila",
            "text": response.text,
            "finish_reason": openai_finish_reason(response.finish_reason),
            "usage": {
                "prompt_tokens": response.prompt_token_count,
                "completion_tokens": response.completion_token_count,
            },
        }

    def format_stream_chunk(self, text: str, done: bool, finish_reason: str = END_TURN) -> str:
        chunk = {"text": text, "done": done}

        if done:
            chunk["finish_reason"] = openai_finish_reason(finish_reason)

        return f"data: {json.dumps(chunk)}\n\n"

    def format_stream_done(self) -> str:
        return "data: [DONE]\n\n"


class MilaChatAdapter(MilaCompletions, ProtocolAdapter):

    def chat_endpoints(self) -> list[ChatEndpoint]:
        return [MilaChat()]
