"""
Mila-native protocol adapter — minimal, no ceremony.
  Chat:        POST /v1/chat/completions
  Completions: POST /v1/completions
"""
import json
import time
import uuid

from schemas.internal import InferenceRequest, InferenceResponse
from protocols.base import ProtocolAdapter
from prompt import build_instruct_prompt


class MilaAdapter(ProtocolAdapter):

    @property
    def chat_path(self) -> str:
        return "/v1/chat/completions"

    @property
    def completions_path(self) -> str:
        return "/v1/completions"

    def parse_chat_request(self, body: dict) -> tuple[str, InferenceRequest]:
        messages = body.get("messages", [])
        system_prompt = body.get("system_prompt", "You are a helpful assistant.")

        history = [m for m in messages[:-1]]
        user_message = messages[-1]["content"] if messages else ""

        prompt_str = build_instruct_prompt(user_message, system_prompt, history)

        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_new_tokens", 256),
            temperature=body.get("temperature", 1.0),
            top_k=body.get("top_k", 0),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]:
        prompt_str = body.get("prompt", "")
        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_new_tokens", 256),
            temperature=body.get("temperature", 1.0),
            top_k=body.get("top_k", 0),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def format_chat_response(self, response: InferenceResponse) -> dict:
        return {
            "id": f"mila-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mila",
            "text": response.text,
            "finish_reason": response.finish_reason,
            "usage": {
                "prompt_tokens": response.prompt_token_count,
                "completion_tokens": response.completion_token_count,
            },
        }

    def format_completions_response(self, response: InferenceResponse) -> dict:
        return {
            "id": f"mila-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "mila",
            "text": response.text,
            "finish_reason": response.finish_reason,
            "usage": {
                "prompt_tokens": response.prompt_token_count,
                "completion_tokens": response.completion_token_count,
            },
        }

    def format_stream_chunk(self, text: str, done: bool) -> str:
        chunk = {
            "text": text,
            "done": done,
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_stream_done(self) -> str:
        return "data: [DONE]\n\n"
