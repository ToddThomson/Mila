"""
OpenAI-compatible protocol adapter.
  Chat:        POST /v1/chat/completions
  Completions: POST /v1/completions
"""
import json
import time
import uuid

from schemas.internal import InferenceRequest, InferenceResponse
from protocols.base import ProtocolAdapter
from prompt import build_instruct_prompt


class OpenAIAdapter(ProtocolAdapter):

    @property
    def chat_path(self) -> str:
        return "/v1/chat/completions"

    @property
    def completions_path(self) -> str:
        return "/v1/completions"

    def parse_chat_request(self, body: dict) -> tuple[str, InferenceRequest]:
        messages = body.get("messages", [])
        system_prompt = body.get("system_prompt", "You are a helpful assistant.")

        # OpenAI convention: system role may appear as first message
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        history = [m for m in messages[:-1]]
        user_message = messages[-1]["content"] if messages else ""

        prompt_str = build_instruct_prompt(user_message, system_prompt, history)

        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_tokens", 256),
            temperature=body.get("temperature", 1.0),
            top_k=body.get("top_k", 0),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]:
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
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mila",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text,
                    },
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.prompt_token_count,
                "completion_tokens": response.completion_token_count,
                "total_tokens": response.prompt_token_count + response.completion_token_count,
            },
        }

    def format_completions_response(self, response: InferenceResponse) -> dict:
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "mila",
            "choices": [
                {
                    "text": response.text,
                    "index": 0,
                    "finish_reason": response.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.prompt_token_count,
                "completion_tokens": response.completion_token_count,
                "total_tokens": response.prompt_token_count + response.completion_token_count,
            },
        }

    def format_stream_chunk(self, text: str, done: bool) -> str:
        chunk = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": "stop" if done else None,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_stream_done(self) -> str:
        return "data: [DONE]\n\n"
