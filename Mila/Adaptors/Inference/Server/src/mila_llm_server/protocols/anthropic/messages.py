"""
Anthropic Messages API protocol adapter.
  Chat:        POST /v1/messages
  Completions: POST /v1/messages  (Anthropic has no separate completions path;
               we alias it to the same endpoint for factory symmetry.)

Gemma family: native tool calling. Inbound tool defs, assistant tool_use blocks,
and user tool_result blocks are mapped onto the same gemma_protocol grammar the
Responses path uses, so Claude Code can drive MIS as a harness. Non-gemma (Llama)
stays tool-blind plain text. Non-streaming tool_use is wired here; streaming
tool_use (content_block_start{type:tool_use} + input_json_delta) is deferred.
"""
import json
import logging
import uuid

from mila_llm_server.schemas.internal import InferenceRequest, InferenceResponse
from mila_llm_server.protocols.base import ProtocolAdapter
from mila_llm_server.protocols.utils import DEFAULT_SYSTEM_PROMPT, extract_content
from mila_llm_server.prompt import build_instruct_prompt
from mila_llm_server.config import settings, loaded, ModelFamily
from mila_llm_server import gemma_protocol

logger = logging.getLogger(__name__)


def _summarize_messages(messages: list) -> list:
    """Compact role/block-type/preview view of an inbound messages array for logging."""
    summary = []

    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        parts = []

        for block in blocks:
            if not isinstance(block, dict):
                parts.append("?")
                continue

            btype = block.get("type", "text")

            if btype == "text":
                parts.append(f"text:{block.get('text', '')[:60]!r}")
            elif btype == "tool_use":
                parts.append(f"tool_use:{block.get('name')}({json.dumps(block.get('input', {}))[:80]})")
            elif btype == "tool_result":
                body = block.get("content", "")
                body = body if isinstance(body, str) else json.dumps(body)
                parts.append(f"tool_result[{block.get('tool_use_id', '')[:12]}]:{body[:80]!r}")
            else:
                parts.append(btype)

        summary.append({role: parts})

    return summary


class AnthropicMessagesAdapter(ProtocolAdapter):

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
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    def _extract_system(self, system) -> str:
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            return " ".join(
                block["text"]
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return DEFAULT_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Request parsing
    # ------------------------------------------------------------------

    def parse_chat_request(self, body: dict) -> tuple[str, InferenceRequest]:
        system_prompt = self._extract_system(body.get("system", DEFAULT_SYSTEM_PROMPT))
        messages = body.get("messages", [])
        tools = body.get("tools") or []

        # Bring-up: log what Claude Code POSTED back -- the inbound message shape
        # (roles + content-block types + previews). Reveals whether the prior
        # assistant tool_use and the user's tool_result arrived well-formed before
        # MIS reassembles them into the Gemma prompt.
        logger.info(
            "anthropic /v1/messages: %d msgs, %d tools; %s",
            len(messages), len(tools), _summarize_messages(messages))

        if loaded.family == ModelFamily.gemma:
            prompt_str = self._build_gemma_prompt(messages, system_prompt, tools)
        else:
            prompt_str = self._build_llama_prompt(messages, system_prompt)

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

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _normalize_tools(self, tools: list) -> list:
        """
        Adapt Anthropic tool defs ({name, description, input_schema}) to the
        OpenAI-ish shape gemma_protocol.build_tool_injection consumes (it reads
        `parameters`, not Anthropic's `input_schema`).
        """
        normalized = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            name = tool.get("name")
            if not name:
                continue

            normalized.append({
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            })

        return normalized

    def _tool_result_text(self, content) -> str:
        """Anthropic tool_result.content is a string or a list of content blocks."""
        if isinstance(content, str):
            return content

        return extract_content(content) if isinstance(content, list) else ""

    def _build_gemma_prompt(self, messages: list, system_prompt: str, tools: list) -> str:
        """
        Assemble a native Gemma agentic prompt from Anthropic messages. Assistant
        `tool_use` and user `tool_result` content blocks replay as
        <|tool_call>/<|tool_response> spans inside a model turn; a conversation that
        ends on a tool result leaves that model turn open so the model continues it
        (mirrors responses.py / the chat harness splice-and-resume).
        """
        tool_injection = gemma_protocol.build_tool_injection(self._normalize_tools(tools))
        # Bring-up: the tool declarations land in the SYSTEM turn (front of the
        # prompt), so they never appear in the _dispatch tail log -- log length +
        # head here to confirm they are actually injected, not filtered.
        logger.info(
            "gemma tool injection: %d tools in, %d chars out; head: %s",
            len(tools), len(tool_injection), tool_injection[:400].replace("\n", " "))
        system_block = system_prompt + tool_injection

        turns: list[dict] = []
        pending_names: dict[str, str] = {}  # tool_use.id -> tool name

        def append_model_tool_span(span: str) -> None:
            # Merge into any immediately-preceding model turn (a tool span OR the
            # assistant's preamble text) so a single Anthropic assistant message
            # that carries text + tool_use stays one model turn -- back-to-back
            # <|turn>model turns are off-distribution for Gemma.
            #
            # NO separator: the reference template emits <tool_call|><|tool_response>
            # back-to-back, and these are atomic special tokens -- a newline between
            # them is an extra token the model never saw in training.
            if turns and turns[-1]["role"] == "model":
                turns[-1]["content"] += span
                turns[-1]["tool"] = True
            else:
                turns.append({"role": "model", "content": span, "tool": True})

        def append_text(role: str, text: str) -> None:
            if not text:
                return
            if turns and turns[-1]["role"] == role and not turns[-1].get("tool"):
                turns[-1]["content"] += "\n" + text
            else:
                turns.append({"role": role, "content": text})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]

            for block in blocks:
                if not isinstance(block, dict):
                    continue

                btype = block.get("type", "text")

                if btype == "text":
                    gemma_role = "model" if role == "assistant" else "user"
                    append_text(gemma_role, block.get("text", ""))

                elif btype == "tool_use":
                    name = block.get("name", "tool")
                    pending_names[block.get("id", "")] = name
                    append_model_tool_span(
                        gemma_protocol.format_tool_call(name, json.dumps(block.get("input", {}))))

                elif btype == "tool_result":
                    name = pending_names.get(block.get("tool_use_id", ""), "tool")
                    append_model_tool_span(
                        gemma_protocol.format_tool_response(
                            name, self._tool_result_text(block.get("content", ""))))

        # If the transcript ends on a tool exchange the model must continue that
        # same model turn; otherwise a fresh model turn is primed.
        continue_open = bool(turns) and turns[-1].get("tool", False)
        return gemma_protocol.assemble_prompt(system_block, turns, continue_open)

    def _build_llama_prompt(self, messages: list, system_prompt: str) -> str:
        """Tool-blind plain-text path for the Llama family (unchanged behavior)."""
        user_message = self._extract_text(messages[-1]["content"]) if messages else ""
        history = [
            {"role": m["role"], "content": self._extract_text(m["content"])}
            for m in messages[:-1]
        ]
        return build_instruct_prompt(user_message, system_prompt, history)

    # ------------------------------------------------------------------
    # Model-text reduction (shared with the factory tool-call detection)
    # ------------------------------------------------------------------

    def parse_tool_call_from_text(self, text: str) -> dict | None:
        """Detect a native Gemma tool call in the raw model output (family-aware)."""
        if loaded.family == ModelFamily.gemma:
            return gemma_protocol.parse_tool_call(text)
        return None

    def clean_response_text(self, text: str) -> str:
        """Reduce raw model output to the user-facing answer (Gemma channel-aware)."""
        if loaded.family == ModelFamily.gemma:
            return gemma_protocol.extract_answer(text)
        return text

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    def format_chat_response(self, response: InferenceResponse) -> dict:
        tool_call = self.parse_tool_call_from_text(response.text)

        if tool_call:
            try:
                tool_input = json.loads(tool_call["arguments"])
            except (json.JSONDecodeError, TypeError):
                tool_input = {}

            content = [
                {
                    "type": "tool_use",
                    # Deterministic call_id round-trips as the client's tool_result
                    # tool_use_id, correlating the result back to this call.
                    "id": tool_call["call_id"],
                    "name": tool_call["name"],
                    "input": tool_input,
                }
            ]
            stop_reason = "tool_use"
        else:
            content = [
                {
                    "type": "text",
                    "text": self.clean_response_text(response.text),
                }
            ]
            stop_reason = "end_turn"

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": loaded.name,
            "stop_reason": stop_reason,
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
                "model": loaded.name,
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

    # ------------------------------------------------------------------
    # Streaming tool_use (buffered path -- see factory._stream_buffered_tool)
    # ------------------------------------------------------------------

    def format_stream_message_start(self, prompt_token_count: int) -> str:
        # Open the message but DEFER content_block_start: the block's type
        # (text vs tool_use) is only known once the whole model turn is parsed.
        message_start = {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": loaded.name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_token_count, "output_tokens": 0},
            },
        }
        ping = {"type": "ping"}
        return (
            f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
            f"event: ping\ndata: {json.dumps(ping)}\n\n"
        )

    def format_stream_text_block(self, text: str) -> str:
        start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }
        stop = {"type": "content_block_stop", "index": 0}
        return (
            f"event: content_block_start\ndata: {json.dumps(start)}\n\n"
            f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
            f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n"
        )

    def format_stream_tool_use_block(self, tool_call: dict) -> str:
        # content_block_start carries the tool id+name with empty input; the arguments
        # arrive as one input_json_delta. Gemma emits the whole call at once, so there
        # is no partial JSON to fragment. The deterministic call_id round-trips as the
        # client's tool_result tool_use_id.
        try:
            tool_input = json.loads(tool_call["arguments"])
        except (json.JSONDecodeError, TypeError):
            tool_input = {}

        start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": tool_call["call_id"],
                "name": tool_call["name"],
                "input": {},
            },
        }
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
        }
        stop = {"type": "content_block_stop", "index": 0}
        return (
            f"event: content_block_start\ndata: {json.dumps(start)}\n\n"
            f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
            f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n"
        )

    def format_stream_message_stop_delta(self, output_token_count: int, stop_reason: str) -> str:
        data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_token_count},
        }
        return f"event: message_delta\ndata: {json.dumps(data)}\n\n"

    def format_responses_stream_keepalive(self, response_id: str) -> str:
        return "event: ping\ndata: {}\n\n"
