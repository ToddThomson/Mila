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

    def _extract_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block["text"]
                for block in content
                if block.get("type") == "text"
            )
        return ""

    def _extract_system(self, system) -> str:
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            return " ".join(
                block["text"]
                for block in system
                if block.get("type") == "text"
            )
        return "You are a helpful assistant."

    def parse_chat_request(self, body: dict) -> tuple[str, InferenceRequest]:

        import json
        print( "[MIS DEBUG] Request: ", json.dumps( body, indent=2))

        # Anthropic puts system at the top level, not inside messages.
        system_prompt = self._extract_system(body.get("system", "You are a helpful assistant."))
        messages = body.get("messages", [])

        user_message = self._extract_text(messages[-1]["content"]) if messages else ""

        history = [
            {"role": m["role"], "content": self._extract_text(m["content"])}
            for m in messages[:-1]
        ]

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

    def format_stream_preamble(self, prompt_token_count: int) -> str:
        message_start = {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "mila",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_token_count, "output_tokens": 0},
            },
        }
        content_block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        ping = {"type": "ping"}
        return (
            f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
            f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
            f"event: ping\ndata: {json.dumps(ping)}\n\n"
        )

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

    def format_stream_message_delta(self, output_token_count: int) -> str:
        data = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_token_count},
        }
        return f"event: message_delta\ndata: {json.dumps(data)}\n\n"

    def format_stream_done(self) -> str:
        message_stop = {"type": "message_stop"}
        return f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"