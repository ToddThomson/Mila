"""
OpenAI-compatible protocol adapter.
  Chat:        POST /v1/chat/completions
  Completions: POST /v1/completions
  Responses:   POST /v1/responses
"""
import json
import time
import uuid

from schemas.internal import InferenceRequest, InferenceResponse
from protocols.base import ProtocolAdapter, ResponsesCapable
from prompt import build_instruct_prompt
from config import settings

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_MODEL_NAME = settings.model_name

def _extract_content(content: str | list) -> str:
    if isinstance(content, str):
        return content
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("input_text", "text")
    ]
    return "".join(text_parts)

class OpenAIAdapter(ProtocolAdapter, ResponsesCapable):

    @property
    def chat_path(self) -> str:
        return "/v1/chat/completions"

    @property
    def completions_path(self) -> str:
        return "/v1/completions"

    @property
    def responses_path(self) -> str:
        return "/v1/responses"

    def parse_chat_request(self, body: dict) -> tuple[str, InferenceRequest]:
        import json
        print("[MIS DEBUG] parse_chat_request:", json.dumps(body, indent=2))

        messages = body.get("messages", [])
        system_prompt = _DEFAULT_SYSTEM_PROMPT

        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        history = messages[:-1]
        user_message = messages[-1]["content"] if messages else ""
        tools = body.get("tools") or None

        prompt_str = build_instruct_prompt(user_message, system_prompt, history, tools)

        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_tokens", settings.default_max_new_tokens),
            temperature=body.get("temperature", settings.default_temperature),
            top_k=body.get("top_k", settings.default_top_k),
            top_p=body.get("top_p", settings.default_top_p),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]:
        import json
        print("[MIS DEBUG] parse_completions_request:", json.dumps(body, indent=2))

        prompt_str = body.get("prompt", "")
        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_tokens", settings.default_max_new_tokens),
            temperature=body.get("temperature", settings.default_temperature),
            top_k=body.get("top_k", settings.default_top_k),
            top_p=body.get("top_p", settings.default_top_p),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def parse_responses_request(self, body: dict) -> tuple[str, InferenceRequest]:
        import json
        print("[MIS DEBUG] parse_responses_request:", json.dumps(body, indent=2))

        raw_input = body.get("input", "")
        instructions = body.get("instructions", _DEFAULT_SYSTEM_PROMPT)
        tools = None # Ignoring tools for now during testing
        # tools = body.get("tools") or None

        if isinstance(raw_input, str):
            user_message = raw_input
            history = []
        else:
            messages = list(raw_input)
            system_parts: list[str] = []

            if instructions and instructions != _DEFAULT_SYSTEM_PROMPT:
                system_parts.append(instructions)

            # Consume leading developer/system turns into the system prompt.
            while messages and messages[0].get("role") in ("system", "developer"):
                system_parts.append(_extract_content(messages[0].get("content", "")))
                messages = messages[1:]

            if system_parts:
                instructions = "\n\n".join(system_parts)

            # Collapse consecutive same-role user turns by merging their content.
            merged: list[dict] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = _extract_content(msg.get("content", ""))
                if merged and merged[-1]["role"] == role:
                    merged[-1]["content"] += "\n" + content
                else:
                    merged.append({"role": role, "content": content})

            history = merged[:-1]
            user_message = merged[-1]["content"] if merged else ""

        print("[MIS DEBUG] instructions:", instructions[:200])
        print("[MIS DEBUG] user_message:", user_message)

        prompt_str = build_instruct_prompt(user_message, instructions, history, tools)
        print("[MIS DEBUG] prompt_str:\n", prompt_str[:500])

        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_output_tokens", settings.default_max_new_tokens),
            temperature=body.get("temperature", settings.default_temperature),
            top_k=body.get("top_k", settings.default_top_k),
            top_p=body.get("top_p", settings.default_top_p),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def format_chat_response(self, response: InferenceResponse) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _MODEL_NAME,
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
            "model": _MODEL_NAME,
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

    def format_responses_response(self, response: InferenceResponse) -> dict:
        response_id = f"resp-{uuid.uuid4().hex}"
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": _MODEL_NAME,
            "status": "completed",
            "incomplete_details": None,
            "error": None,
            "output": [
                {
                    "id": f"msg-{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response.text,
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": response.prompt_token_count,
                "output_tokens": response.completion_token_count,
                "total_tokens": response.prompt_token_count + response.completion_token_count,
            },
        }

    def format_stream_chunk(self, text: str, done: bool) -> str:
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _MODEL_NAME,
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

    def format_responses_stream_created(self, response_id: str) -> str:
        data = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": _MODEL_NAME,
                "status": "in_progress",
                "incomplete_details": None,
                "error": None,
                "output": [],
                "usage": None,
            },
        }
        return f"event: response.created\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_output_item_added(self, response_id: str, item_id: str) -> str:
        data = {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        }
        return f"event: response.output_item.added\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_content_part_added(self, response_id: str) -> str:
        data = {
            "type": "response.content_part.added",
            "response_id": response_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
            },
        }
        return f"event: response.content_part.added\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_content_part_done(self, response_id: str, text: str) -> str:
        data = {
            "type": "response.content_part.done",
            "response_id": response_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": text,
            },
        }
        return f"event: response.content_part.done\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_output_item_done(self, response_id: str, item_id: str, text: str) -> str:
        data = {
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            },
        }
        return f"event: response.output_item.done\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_chunk(self, text: str, done: bool, response_id: str) -> str:
        if done:
            event_type = "response.output_text.done"
            data = {
                "type": event_type,
                "response_id": response_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            }
        else:
            event_type = "response.output_text.delta"
            data = {
                "type": event_type,
                "response_id": response_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_keepalive(self, response_id: str) -> str:
        data = {
            "type": "response.in_progress",
            "response_id": response_id,
        }
        return f"event: response.in_progress\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_done(self, response_id: str, output_text: str = "") -> str:
        data = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": _MODEL_NAME,
                "status": "completed",
                "incomplete_details": None,
                "error": None,
                "output": [
                    {
                        "id": f"msg-{uuid.uuid4().hex}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": output_text,
                            }
                        ],
                    }
                ],
            },
        }
        return f"event: response.completed\ndata: {json.dumps(data)}\n\n"
