"""
Anthropic Messages API protocol adapter.
  Chat:        POST /v1/messages
  Completions: POST /v1/messages  (Anthropic has no separate completions path;
               we alias it to the same endpoint for factory symmetry.)
"""
import json
import time
import uuid

from schemas.internal import InferenceRequest, InferenceResponse
from protocols.base import ProtocolAdapter
from prompt import build_instruct_prompt
from config import settings


class AnthropicAdapter(ProtocolAdapter):

    @property
    def chat_path(self) -> str:
        return "/v1/messages"

    @property
    def completions_path(self) -> str:
        # Anthropic has no separate completions endpoint; alias to messages.
        return "/v1/messages/completions"

    def parse_chat_request(self, body: dict) -> tuple[str, InferenceRequest]:
        # Anthropic puts system at the top level, not inside messages.
        system_prompt = body.get("system", "You are a helpful assistant.")
        messages = body.get("messages", [])

        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
        user_message = messages[-1]["content"] if messages else ""

        prompt_str = build_instruct_prompt(user_message, system_prompt, history)

        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_tokens", settings.default_max_new_tokens),
            temperature=body.get("temperature", 1.0),
            top_k=body.get("top_k", 0),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]:
        # Treat as a bare user message for simplicity.
        prompt_str = body.get("prompt", "")
        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_tokens", 256),
            temperature=body.get("temperature", 1.0),
            top_k=body.get("top_k", 0),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def format_chat_response(self, response: InferenceResponse) -> dict:
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.text,
                }
            ],
            "model": "mila",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.prompt_token_count,
                "output_tokens": response.completion_token_count,
            },
        }

    def format_completions_response(self, response: InferenceResponse) -> dict:
        return self.format_chat_response(response)

    def format_stream_chunk(self, text: str, done: bool) -> str:
        if done:
            event_type = "content_block_stop"
            data = {"type": "content_block_stop", "index": 0}
        else:
            event_type = "content_block_delta"
            data = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": text,
                },
            }
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def format_stream_done(self) -> str:
        message_stop = {"type": "message_stop"}
        return f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
