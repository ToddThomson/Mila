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


class AnthropicMessages(ChatEndpoint):

    @property
    def path(self) -> str:
        return "/v1/messages"

    def parse(self, body: dict, protocol) -> ServedRequest:
        turns, names_by_id = [], {}
        system = body.get("system")

        if system:
            turns.append({"role": "system", "content": text_of(system, field_name="system")})

        for index, message in enumerate(body.get("messages") or []):
            role = message.get("role")
            where = f"messages[{index}]"
            content = message.get("content", "")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []

            if role == "user":
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
                        turn["reasoning"] = block.get("thinking", "")
                    elif kind == "tool_use":
                        names_by_id[block.get("id", "")] = block.get("name", "")
                        turn["tool_calls"].append({"name": block.get("name", ""), "arguments": json.dumps(block.get("input") or {})})
                    else:
                        raise RequestError(f"{where}.content[{position}] is a '{kind}' block, which this server does not read")

                turns.append(turn)

            else:
                raise RequestError(f"{where}.role '{role}' is not one this server reads")

        variables = {}
        thinking = body.get("thinking")

        if thinking is not None:
            if thinking.get("type") == "disabled":
                name, value = template_variable(protocol, "enable_thinking", "false", field_name="thinking.type")
                variables[name] = value
            else:
                raise RequestError(
                    "thinking with budget_tokens is not supported: this model's template has no thinking budget. "
                    "Send thinking.type 'disabled', or omit thinking for the template's default")

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
        )

    @staticmethod
    def _content(reply: ServedReply) -> list[dict]:
        content = []

        if reply.reasoning:
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
        return self._message(context, self._content(reply), reply)

    def stream_open(self, context: ReplyContext) -> str:
        return _event("message_start", {"message": self._message(context, [], None)}) + _event("ping", {})

    def stream_keepalive(self, context: ReplyContext) -> str:
        return _event("ping", {})

    def stream_reply(self, reply: ServedReply, context: ReplyContext) -> str:
        events = []

        for index, block in enumerate(self._content(reply)):
            if block["type"] == "thinking":
                events.append(_event("content_block_start", {"index": index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
                events.append(_event("content_block_delta", {"index": index, "delta": {"type": "thinking_delta", "thinking": block["thinking"]}}))
                events.append(_event("content_block_delta", {"index": index, "delta": {"type": "signature_delta", "signature": ""}}))
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
