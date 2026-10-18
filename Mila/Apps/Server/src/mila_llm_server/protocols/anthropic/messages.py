"""
Anthropic Messages.
  Chat:        POST /v1/messages

A parser into the conversation and a formatter from the reply, with nothing about which model is loaded
(InferenceServerWire.md 3.3); what it honours and refuses is 3.6's table.
"""
import json

from mila_llm_server.conversation import (
    RequestError,
    ServedReply,
    ServedRequest,
    assistant_turn,
    has_template_variable,
    not_offered,
    offered_tool,
    template_variable,
    text_of,
    tool_turn,
)
from mila_llm_server.protocols.base import ChatEndpoint, ReplyContext
from mila_llm_server.protocols.utils import anthropic_stop_reason, parse_stop


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"


def _tool_choice_offers_tools(tool_choice) -> tuple[bool, bool]:
    """Whether tools are offered, and whether more than one call a turn is allowed."""
    if tool_choice is None:
        return True, True

    kind = tool_choice.get("type")
    parallel = not tool_choice.get("disable_parallel_tool_use", False)

    if kind == "auto":
        return True, parallel

    if kind == "none":
        return False, parallel

    raise RequestError(f"tool_choice {json.dumps(tool_choice)} is not supported: the model cannot be made to call a tool")


def _system_message_text(messages: list[dict], index: int) -> str:
    """
    A system message inside `messages`, placed as Anthropic places one: after a user message, and last or before
    an assistant message (InferenceServerWire.md 3.6).
    """
    message, where = messages[index], f"messages[{index}]"

    if message.get("clear_at", "never") != "never":
        raise RequestError(f"{where}.clear_at '{message['clear_at']}' is not supported by this server")

    if "output_config" in message:
        raise RequestError(f"{where}.output_config is not supported by this server")

    text = text_of(message.get("content"), field_name=f"{where}.content")

    if not text.strip():
        return text

    # Consecutive system messages are placed together, so each is judged by the messages around the run.
    previous = next((other.get("role") for other in reversed(messages[:index]) if other.get("role") != "system"), None)
    following = next((other.get("role") for other in messages[index + 1:] if other.get("role") != "system"), "assistant")

    if not (previous == "user" and following == "assistant"):
        raise RequestError(f"{where} is a system message out of place: it must follow a user message, and be the last "
                           "message or be followed by an assistant message")

    return text


def _context_management_variables(context_management: dict, protocol) -> dict[str, str]:
    """
    Anthropic's context edits the template can make (InferenceServerWire.md 3.6): keeping all past thinking is
    the template's own `preserve_thinking`. Nothing here clears history, so every edit that would is refused.
    """
    variables = {}

    for index, edit in enumerate(context_management.get("edits") or []):
        where = f"context_management.edits[{index}]"

        if edit.get("type") != "clear_thinking_20251015":
            raise RequestError(f"{where} '{edit.get('type')}' is not supported by this server")

        keep = edit.get("keep")

        if keep is not None and keep != "all":
            raise RequestError(f"{where}.keep is not supported other than 'all': no template keeps a count of turns")

        # No keep is Anthropic's per-model default, which here is the template's.
        if keep == "all" and has_template_variable(protocol, "preserve_thinking"):
            name, value = template_variable(protocol, "preserve_thinking", "true", field_name=f"{where}.keep")
            variables[name] = value

    return variables


class AnthropicMessages(ChatEndpoint):

    @property
    def path(self) -> str:
        return "/v1/messages"

    def parse(self, body: dict, protocol) -> ServedRequest:
        turns, names_by_id = [], {}
        system = body.get("system")

        if system:
            turns.append({"role": "system", "content": text_of(system, field_name="system")})

        messages = body.get("messages") or []

        for index, message in enumerate(messages):
            role = message.get("role")
            where = f"messages[{index}]"
            content = message.get("content", "")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []

            if role == "system":
                turns.append({"role": "system", "content": _system_message_text(messages, index)})

            elif role == "user":
                texts = []

                for position, block in enumerate(blocks):
                    kind = block.get("type")

                    if kind == "text":
                        texts.append(block.get("text", ""))
                    elif kind == "tool_result":
                        # A result is the tool's turn, not the user's: rendered as user text it is another
                        # message from the person (InferenceServerWire.md 5).
                        turns.append(tool_turn(
                            names_by_id, block.get("tool_use_id", ""),
                            text_of(block.get("content"), field_name=f"{where}.content[{position}].content"),
                            field_name=f"{where}.content[{position}].tool_use_id"))
                    else:
                        raise RequestError(f"{where}.content[{position}] is a '{kind}' block, which this model cannot be given")

                if texts:
                    turns.append({"role": "user", "content": "\n".join(texts)})

            elif role == "assistant":
                turn = assistant_turn()

                for position, block in enumerate(blocks):
                    kind = block.get("type")

                    if kind == "text":
                        turn["content"] = (turn["content"] + "\n" + block.get("text", "")) if turn["content"] else block.get("text", "")
                    elif kind == "thinking":
                        turn["reasoning"] = block.get("thinking") or block.get("signature") or ""
                    elif kind == "tool_use":
                        names_by_id[block.get("id", "")] = block.get("name", "")
                        turn["tool_calls"].append({"name": block.get("name", ""), "arguments": json.dumps(block.get("input") or {})})
                    else:
                        raise RequestError(f"{where}.content[{position}] is a '{kind}' block, which this server does not read")

                turns.append(turn)

            else:
                raise RequestError(f"{where}.role '{role}' is not one this server reads")

        variables = {}
        thinking = body.get("thinking") or {}
        kind = thinking.get("type")

        if kind in ("adaptive", "disabled"):
            # A template with no switch never thinks, which meets both: adaptive leaves whether to the model.
            if has_template_variable(protocol, "enable_thinking"):
                name, value = template_variable(
                    protocol, "enable_thinking", "true" if kind == "adaptive" else "false", field_name="thinking.type")
                variables[name] = value

        elif kind is not None:
            raise RequestError(
                f"thinking.type '{kind}' is not supported: this model's template has no thinking budget. "
                "Send 'adaptive' or 'disabled', or omit thinking for the template's default")

        display = thinking.get("display")

        if display not in (None, "summarized", "omitted"):
            raise RequestError(f"thinking.display '{display}' is not supported by this server")

        output_config = body.get("output_config") or {}

        for key in output_config:
            if key == "effort":
                name, value = template_variable(protocol, "reasoning_effort", output_config["effort"], field_name="output_config.effort")
                variables[name] = value
            elif key == "format":
                raise RequestError("output_config.format is not supported: there is no constrained decoding")
            else:
                raise RequestError(f"output_config.{key} is not supported by this server")

        variables.update(_context_management_variables(body.get("context_management") or {}, protocol))

        offers_tools, parallel = _tool_choice_offers_tools(body.get("tool_choice"))
        tools = []

        if offers_tools:
            for tool in body.get("tools") or []:
                if tool.get("type") not in (None, "custom"):
                    not_offered(tool.get("type"), tool.get("name"))
                    continue

                declared = offered_tool(tool.get("name"), tool.get("description"), tool.get("input_schema"))

                if declared is not None:
                    tools.append(declared)

        return ServedRequest(
            turns=turns,
            tools=tools,
            template_variables=variables,
            max_new_tokens=body.get("max_tokens"),
            temperature=body.get("temperature"),
            top_k=body.get("top_k"),
            top_p=body.get("top_p"),
            stop=parse_stop(body.get("stop_sequences")),
            stream=bool(body.get("stream", False)),
            parallel_tool_calls=parallel,
            wire={"omit_thinking": display == "omitted"},
        )

    @staticmethod
    def _content(reply: ServedReply, context: ReplyContext) -> list[dict]:
        content = []

        if reply.reasoning:
            # Omitted thinking travels in the signature, as Anthropic's does, so the block a client sends back
            # still carries what the model wrote (InferenceServerWire.md 3.6).
            if context.wire.get("omit_thinking"):
                content.append({"type": "thinking", "thinking": "", "signature": reply.reasoning})
            else:
                content.append({"type": "thinking", "thinking": reply.reasoning, "signature": ""})

        if reply.content or not reply.tool_calls:
            content.append({"type": "text", "text": reply.content})

        for call in reply.tool_calls:
            content.append({"type": "tool_use", "id": call.call_id, "name": call.name, "input": json.loads(call.arguments or "{}")})

        return content

    @staticmethod
    def _message(context: ReplyContext, content: list[dict], reply: ServedReply | None) -> dict:
        return {
            "id": context.response_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": context.model,
            "stop_reason": None if reply is None else anthropic_stop_reason(reply.finish_reason),
            "stop_sequence": None if reply is None else reply.stop_sequence,
            "usage": {
                "input_tokens": context.prompt_tokens,
                "output_tokens": 0 if reply is None else reply.completion_tokens,
            },
        }

    def format(self, reply: ServedReply, context: ReplyContext) -> dict:
        return self._message(context, self._content(reply, context), reply)

    def stream_open(self, context: ReplyContext) -> str:
        return _event("message_start", {"message": self._message(context, [], None)}) + _event("ping", {})

    def stream_keepalive(self, context: ReplyContext) -> str:
        return _event("ping", {})

    def stream_error(self, message: str, context: ReplyContext) -> str:
        return _event("error", {"error": {"type": "api_error", "message": message}})

    def stream_reply(self, reply: ServedReply, context: ReplyContext) -> str:
        events = []

        for index, block in enumerate(self._content(reply, context)):
            if block["type"] == "thinking":
                events.append(_event("content_block_start", {"index": index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
                events.append(_event("content_block_delta", {"index": index, "delta": {"type": "thinking_delta", "thinking": block["thinking"]}}))
                events.append(_event("content_block_delta", {"index": index, "delta": {"type": "signature_delta", "signature": block["signature"]}}))
            elif block["type"] == "text":
                events.append(_event("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}}))
                events.append(_event("content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": block["text"]}}))
            else:
                events.append(_event("content_block_start", {"index": index, "content_block": {**block, "input": {}}}))
                events.append(_event("content_block_delta", {"index": index, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))

            events.append(_event("content_block_stop", {"index": index}))

        events.append(_event("message_delta", {
            "delta": {"stop_reason": anthropic_stop_reason(reply.finish_reason), "stop_sequence": reply.stop_sequence},
            "usage": {"output_tokens": reply.completion_tokens}}))
        events.append(_event("message_stop", {}))

        return "".join(events)
