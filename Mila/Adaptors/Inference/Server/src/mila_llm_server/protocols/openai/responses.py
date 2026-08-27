"""
OpenAI Responses API protocol adapter.
  Responses:   POST /v1/responses
"""
import json
import time
import uuid

from mila_llm_server.schemas.internal import InferenceRequest, InferenceResponse
from mila_llm_server.protocols.base import ResponsesCapable
from mila_llm_server.protocols.utils import DEFAULT_SYSTEM_PROMPT, extract_content
from mila_llm_server.protocols.openai.tool_bridge import build_tool_injection, parse_tool_call
from mila_llm_server.prompt import build_instruct_prompt
from mila_llm_server.config import settings, loaded, ModelFamily
from mila_llm_server import gemma_protocol, qwen_bridge

import mila


class OpenAIResponsesAdapter(ResponsesCapable):

    @property
    def responses_path(self) -> str:
        return "/v1/responses"

    def parse_responses_request(self, body: dict) -> tuple[str, InferenceRequest]:
        raw_input = body.get("input", "")
        instructions = body.get("instructions", DEFAULT_SYSTEM_PROMPT)
        tools = body.get("tools") or []

        # Bring-up: log the tool schemas + input item shapes Codex sends so we
        # can see what is a first-class function tool vs a built-in, and how
        # apply_patch/list_dir arrive. Trim once the tool loop is settled.
        _tool_names = [(t.get("name") or (t.get("function") or {}).get("name"), t.get("type")) for t in tools]
        print("[MIS DEBUG] tools:", json.dumps(_tool_names), flush=True)
        if not isinstance(raw_input, str):
            _input_shapes = [(m.get("type", "message"), m.get("role")) for m in raw_input]
            print("[MIS DEBUG] input items:", json.dumps(_input_shapes), flush=True)

        # Three-way, not gemma-or-else: the else branch was Llama's tool bridge back when
        # Llama was the only other family, and it splices Llama 3's <|python_tag|> grammar
        # into the prompt. A Qwen model sent through it would be handed a grammar it was
        # never trained to read.
        if loaded.family == ModelFamily.gemma:
            prompt_str = self._build_gemma_prompt(raw_input, instructions, tools)
        elif loaded.family == ModelFamily.qwen:
            prompt_str = self._build_qwen_prompt(raw_input, instructions, tools)
        elif loaded.family == ModelFamily.llama:
            prompt_str = self._build_llama_prompt(raw_input, instructions, tools)
        else:
            prompt_str = self._build_plain_prompt(raw_input, instructions)

        print("[MIS DEBUG] prompt_str:\n", prompt_str, flush=True)

        req = InferenceRequest(
            prompt_ids=[],
            max_new_tokens=body.get("max_output_tokens", settings.default_max_new_tokens),
            temperature=body.get("temperature", settings.default_temperature),
            top_k=body.get("top_k", settings.default_top_k),
            top_p=body.get("top_p", settings.default_top_p),
            stream=body.get("stream", False),
        )
        return prompt_str, req

    def _collect_system_block(self, messages: list, instructions: str) -> tuple[str, list]:
        """Fold leading system/developer items + instructions into one system block."""
        system_parts: list[str] = []

        if instructions and instructions != DEFAULT_SYSTEM_PROMPT:
            system_parts.append(instructions)

        while messages and messages[0].get("role") in ("system", "developer"):
            system_parts.append(extract_content(messages[0].get("content", "")))
            messages = messages[1:]

        system_block = "\n\n".join(system_parts) if system_parts else instructions
        return system_block, messages

    def _build_gemma_prompt(self, raw_input, instructions: str, tools: list) -> str:
        """
        Assemble a native Gemma agentic prompt. Function-call / function-call-output
        items replay as <|tool_call>/<|tool_response> spans inside a model turn;
        a conversation that ends on a tool result leaves that model turn open so
        the model continues it (mirrors the chat harness splice-and-resume).
        """
        if isinstance(raw_input, str):
            system_block = instructions + gemma_protocol.build_tool_injection(tools)
            turns = [{"role": "user", "content": raw_input}]
            return gemma_protocol.assemble_prompt(system_block, turns, continue_open=False)

        messages = list(raw_input)
        system_block, messages = self._collect_system_block(messages, instructions)
        system_block += gemma_protocol.build_tool_injection(tools)

        turns: list[dict] = []
        pending_names: dict[str, str] = {}  # call_id -> tool name

        def append_model_tool_span(span: str) -> None:
            # Spans concatenate with NO separator: the reference template emits
            # <tool_call|><|tool_response> back-to-back, and these are atomic special
            # tokens -- a newline between them is an extra token the model never saw.
            if turns and turns[-1]["role"] == "model" and turns[-1].get("tool"):
                turns[-1]["content"] += span
            else:
                turns.append({"role": "model", "content": span, "tool": True})

        for msg in messages:
            msg_type = msg.get("type", "message")

            if msg_type == "function_call":
                name = msg.get("name", "tool")
                pending_names[msg.get("call_id", "")] = name
                append_model_tool_span(
                    gemma_protocol.format_tool_call(name, msg.get("arguments", "{}")))

            elif msg_type == "function_call_output":
                name = pending_names.get(msg.get("call_id", ""), "tool")
                raw_output = msg.get("output", "")
                # Bring-up: see the exact envelope Codex sends so we know which
                # field is the real output vs metadata (e.g. the leaked chunk id).
                print("[MIS DEBUG] tool_output:", repr(raw_output)[:600], flush=True)
                append_model_tool_span(
                    gemma_protocol.format_tool_response(name, raw_output))

            elif msg_type == "message":
                role = "model" if msg.get("role") == "assistant" else "user"
                content = extract_content(msg.get("content", ""))
                if turns and turns[-1]["role"] == role and not turns[-1].get("tool"):
                    turns[-1]["content"] += "\n" + content
                else:
                    turns.append({"role": role, "content": content})

        # If the transcript ends on a tool exchange the model must continue that
        # same turn; otherwise the last turn is the user's message and a fresh
        # model turn is primed.
        continue_open = bool(turns) and turns[-1].get("tool", False)
        return gemma_protocol.assemble_prompt(system_block, turns, continue_open)

    def _build_llama_prompt(self, raw_input, instructions: str, tools: list) -> str:
        if isinstance(raw_input, str):
            user_message = raw_input
            history = []
        else:
            messages = list(raw_input)
            system_block, messages = self._collect_system_block(messages, instructions)
            instructions = system_block

            merged: list[dict] = []
            pending_calls: dict[str, str] = {}

            for msg in messages:
                msg_type = msg.get("type", "message")
                role = msg.get("role", "user")

                if msg_type == "function_call":
                    call_id = msg.get("call_id", "")
                    name = msg.get("name", "tool")
                    arguments = msg.get("arguments", "{}")
                    pending_calls[call_id] = name
                    merged.append({
                        "role": "assistant",
                        "content": f"<|python_tag|>[{name}(cmd={arguments})]",
                    })

                elif msg_type == "function_call_output":
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

        return build_instruct_prompt(user_message, instructions, history, None)

    def _build_qwen_prompt(self, raw_input, instructions: str, tools: list) -> str:
        """
        Assemble a native Qwen agentic prompt from Responses items.

        No template is written here -- the turns go to `mila.qwen_format_prompt`, the runtime's
        own renderer. A `function_call` item becomes a tool call on the assistant turn and a
        `function_call_output` becomes a turn of its own, which the template renders as a user
        turn carrying a tool_response span.
        """
        if isinstance(raw_input, str):
            turns = [{"role": "user", "content": raw_input}]
            system_block = instructions
        else:
            messages = list(raw_input)
            system_block, messages = self._collect_system_block(messages, instructions)
            turns = []

            for msg in messages:
                msg_type = msg.get("type", "message")

                if msg_type == "function_call":
                    call = {
                        "id": msg.get("call_id", ""),
                        "name": msg.get("name", "tool"),
                        "arguments": msg.get("arguments", "{}"),
                    }

                    if turns and turns[-1]["role"] == "assistant":
                        turns[-1].setdefault("tool_calls", []).append(call)
                    else:
                        turns.append({"role": "assistant", "content": "", "tool_calls": [call]})

                elif msg_type == "function_call_output":
                    turns.append({
                        "role": "tool",
                        "content": msg.get("output", "") or "(no output)",
                    })

                elif msg_type == "message":
                    role = "assistant" if msg.get("role") == "assistant" else "user"
                    content = extract_content(msg.get("content", ""))

                    if turns and turns[-1]["role"] == role and not turns[-1].get("tool_calls"):
                        turns[-1]["content"] += "\n" + content
                    else:
                        turns.append({"role": role, "content": content})

        if system_block:
            turns.insert(0, {"role": "system", "content": system_block})

        return mila.qwen_format_prompt(
            turns,
            enable_thinking=False,
            tools_json=json.dumps(tools) if tools else "",
        )

    def _build_plain_prompt(self, raw_input, instructions: str) -> str:
        """
        Tool-blind assembly for a family with no agentic grammar on this path yet.

        Tool declarations are dropped and a tool result replays as ordinary user text, so
        the conversation stays readable and nothing invents a call syntax the model was not
        trained on. The family's own turn template still applies -- build_instruct_prompt
        dispatches on it -- so this is a plain conversation in the right frame, not a
        Llama prompt wearing another model's name.
        """
        if isinstance(raw_input, str):
            return build_instruct_prompt(raw_input, instructions, [], None)

        messages = list(raw_input)
        system_block, messages = self._collect_system_block(messages, instructions)

        merged: list[dict] = []

        for msg in messages:
            msg_type = msg.get("type", "message")
            role = msg.get("role", "user")

            if msg_type == "function_call":
                name = msg.get("name", "tool")
                content = f"Called {name} with {msg.get('arguments', '{}')}."
                role = "assistant"

            elif msg_type == "function_call_output":
                content = f"Result: {msg.get('output', '') or '(no output)'}"
                role = "user"

            elif msg_type == "message":
                content = extract_content(msg.get("content", ""))

            else:
                continue

            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += "\n" + content
            else:
                merged.append({"role": role, "content": content})

        user_message = merged[-1]["content"] if merged else ""

        return build_instruct_prompt(user_message, system_block, merged[:-1], None)

    def parse_tool_call_from_text(self, text: str) -> dict | None:
        """Expose the tool-call parser to the streaming factory path (family-aware)."""
        if loaded.family == ModelFamily.gemma:
            return gemma_protocol.parse_tool_call(text)

        if loaded.family == ModelFamily.qwen:
            return qwen_bridge.parse_tool_call(text)

        # Llama's bridge only. A family whose prompt carried no call syntax cannot have
        # emitted one, and running a parser over its prose can only produce a phantom.
        if loaded.family == ModelFamily.llama:
            return parse_tool_call(text)

        return None

    def clean_response_text(self, text: str) -> str:
        """Reduce raw model output to the user-facing answer."""
        if loaded.family == ModelFamily.gemma:
            return gemma_protocol.extract_answer(text)

        if loaded.family == ModelFamily.qwen:
            return qwen_bridge.answer_text(text)

        return text

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
                "model": loaded.name,
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

        tool_call_item = self.parse_tool_call_from_text(response.text)
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
                            "text": self.clean_response_text(response.text),
                        }
                    ],
                }
            ]
            status = "completed"

        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": loaded.name,
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
                "model": loaded.name,
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
                "model": loaded.name,
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
