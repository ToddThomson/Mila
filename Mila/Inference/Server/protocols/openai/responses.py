"""
OpenAI Responses API protocol adapter.
  Responses:   POST /v1/responses
"""
import json
import time
import uuid

from schemas.internal import InferenceRequest, InferenceResponse
from protocols.base import ResponsesCapable
from protocols.utils import DEFAULT_SYSTEM_PROMPT, MODEL_NAME, extract_content
from protocols.openai.tool_bridge import build_tool_injection, parse_tool_call
from prompt import build_instruct_prompt
from config import settings


class OpenAIResponsesAdapter(ResponsesCapable):

    @property
    def responses_path(self) -> str:
        return "/v1/responses"

    def parse_responses_request(self, body: dict) -> tuple[str, InferenceRequest]:
        # DEBUG print("[MIS DEBUG] parse_responses_request:", json.dumps(body, indent=2))

        raw_input = body.get("input", "")
        instructions = body.get("instructions", DEFAULT_SYSTEM_PROMPT)
        tools = body.get("tools") or []

        if isinstance(raw_input, str):
            user_message = raw_input
            history = []
        else:
            messages = list(raw_input)
            system_parts: list[str] = []

            if instructions and instructions != DEFAULT_SYSTEM_PROMPT:
                system_parts.append(instructions)

            while messages and messages[0].get("role") in ("system", "developer"):
                system_parts.append(extract_content(messages[0].get("content", "")))
                messages = messages[1:]

            if system_parts:
                instructions = "\n\n".join(system_parts)

            # Build history, converting function_call/function_call_output items
            # into Llama tool turns so the model sees the result and can conclude.
            merged: list[dict] = []
            pending_calls: dict[str, str] = {}  # call_id -> tool name

            for msg in messages:
                msg_type = msg.get("type", "message")
                role = msg.get("role", "user")

                if msg_type == "function_call":
                    # Assistant emitted a tool call — record for matching output.
                    call_id = msg.get("call_id", "")
                    name = msg.get("name", "tool")
                    arguments = msg.get("arguments", "{}")
                    pending_calls[call_id] = name
                    merged.append({
                        "role": "assistant",
                        "content": f"<|python_tag|>[{name}(cmd={arguments})]",
                    })

                elif msg_type == "function_call_output":
                    # Tool result — present as a tool role message.
                    call_id = msg.get("call_id", "")
                    output = msg.get("output", "")
                    merged.append({
                        "role": "tool",
                        "content": output if output else "(no output)",
                    })

                elif msg_type == "message":
                    content = extract_content(msg.get("content", ""))
                    if merged and merged[-1]["role"] == role:
                        merged[-1]["content"] += "\n" + content
                    else:
                        merged.append({"role": role, "content": content})

            history = merged[:-1]
            user_message = merged[-1]["content"] if merged else ""

        # Only inject tool descriptions when there are no tool results yet —
        # once the model has seen a result it should conclude, not call again.
        has_tool_result = any(
            m.get("type") == "function_call_output"
            for m in (raw_input if not isinstance(raw_input, str) else [])
        )

        if not has_tool_result:
            tool_injection = build_tool_injection(tools)
            if tool_injection:
                instructions = instructions + tool_injection
        else:
            instructions = instructions + (
                "\n\nThe tool has been executed and the result is in the conversation. "
                "Summarize what was done in one short sentence. Do NOT emit another tool call."
            )

        # DEBUG: print("[MIS DEBUG] instructions:", instructions[:200])
        # DEBUG: print("[MIS DEBUG] user_message:", user_message)

        prompt_str = build_instruct_prompt(user_message, instructions, history, None)
        print("[MIS DEBUG] prompt_str:\n", prompt_str)

        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_output_tokens", settings.default_max_new_tokens),
            temperature=body.get("temperature", settings.default_temperature),
            top_k=body.get("top_k", settings.default_top_k),
            top_p=body.get("top_p", settings.default_top_p),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def parse_tool_call_from_text(self, text: str) -> dict | None:
        """Expose parse_tool_call for use by the streaming factory path."""
        return parse_tool_call(text)

    def format_responses_stream_function_call(self, response_id: str, item: dict) -> str:
        item_added = {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": item["id"],
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": "",
            },
        }
        args_delta = {
            "type": "response.function_call_arguments.delta",
            "response_id": response_id,
            "item_id": item["id"],
            "output_index": 0,
            "call_id": item["call_id"],
            "delta": item["arguments"],
        }
        args_done = {
            "type": "response.function_call_arguments.done",
            "response_id": response_id,
            "item_id": item["id"],
            "output_index": 0,
            "call_id": item["call_id"],
            "arguments": item["arguments"],
        }
        item_done = {
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": item["id"],
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": item["arguments"],
            },
        }
        return (
            f"event: response.output_item.added\ndata: {json.dumps(item_added)}\n\n"
            f"event: response.function_call_arguments.delta\ndata: {json.dumps(args_delta)}\n\n"
            f"event: response.function_call_arguments.done\ndata: {json.dumps(args_done)}\n\n"
            f"event: response.output_item.done\ndata: {json.dumps(item_done)}\n\n"
        )

    def format_responses_stream_done_with_tool_call(self, response_id: str, item: dict) -> str:
        data = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": MODEL_NAME,
                "status": "completed",
                "incomplete_details": None,
                "error": None,
                "output": [
                    {
                        "type": "function_call",
                        "id": item["id"],
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "arguments": item["arguments"],
                    }
                ],
            },
        }
        return f"event: response.completed\ndata: {json.dumps(data)}\n\n"

    def format_responses_response(self, response: InferenceResponse) -> dict:
        response_id = f"resp-{uuid.uuid4().hex}"

        tool_call_item = parse_tool_call(response.text)
        if tool_call_item:
            output = [tool_call_item]
            status = "completed"
        else:
            output = [
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
            ]
            status = "completed"

        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": MODEL_NAME,
            "status": status,
            "incomplete_details": None,
            "error": None,
            "output": output,
            "usage": {
                "input_tokens": response.prompt_token_count,
                "output_tokens": response.completion_token_count,
                "total_tokens": response.prompt_token_count + response.completion_token_count,
            },
        }

    def format_responses_stream_created(self, response_id: str) -> str:
        data = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": MODEL_NAME,
                "status": "in_progress",
                "incomplete_details": None,
                "error": None,
                "output": [],
                "usage": None,
            },
        }
        return f"event: response.created\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_keepalive(self, response_id: str) -> str:
        data = {
            "type": "response.in_progress",
            "response_id": response_id,
        }
        return f"event: response.in_progress\ndata: {json.dumps(data)}\n\n"

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

    def format_responses_stream_done(self, response_id: str, output_text: str = "") -> str:
        data = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": MODEL_NAME,
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
