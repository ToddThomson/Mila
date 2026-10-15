"""
OpenAI Responses.
  Responses:   POST /v1/responses

A parser into the conversation and a formatter from the reply, with nothing about which model is loaded
(InferenceServerWire.md 3.3); what it honours and refuses is 3.6's table. The server is stateless: every
request carries the whole transcript, reasoning included, and nothing is kept between them.
"""
import json
import uuid

from mila_llm_server.conversation import (
    RequestError,
    ServedReply,
    ServedRequest,
    assistant_turn,
    not_offered,
    offered_tool,
    reasoning_effort_variables,
    text_of,
    tool_turn,
)
from mila_llm_server.protocols.base import ChatEndpoint, ReplyContext
from mila_llm_server.protocols.openai.chat import tool_choice_offers_tools
from mila_llm_server.protocols.utils import CONTEXT_LIMIT, MAX_TOKENS

#: Codex asks for this so its stateless loop can send reasoning back; it is honoured by carrying the text.
_ENCRYPTED_REASONING = "reasoning.encrypted_content"


def _status(reply: ServedReply) -> tuple[str, dict | None]:
    """A reply cut at its token budget or the context is `incomplete`, as OpenAI reports one."""
    if reply.finish_reason in (MAX_TOKENS, CONTEXT_LIMIT):
        return "incomplete", {"reason": "max_output_tokens"}

    return "completed", None


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"


def _reasoning_text(item: dict) -> str:
    summary = "".join(part.get("text", "") for part in item.get("summary") or [] if isinstance(part, dict))

    return summary or item.get("encrypted_content") or ""


class OpenAIResponses(ChatEndpoint):

    @property
    def path(self) -> str:
        return "/v1/responses"

    def parse(self, body: dict, protocol) -> ServedRequest:
        if body.get("previous_response_id"):
            raise RequestError("previous_response_id is not supported: the server keeps no state between requests")

        if body.get("store") is True:
            raise RequestError("store is not supported: the server keeps no state between requests")

        text_format = (body.get("text") or {}).get("format") or {}

        if text_format.get("type", "text") != "text":
            raise RequestError("text.format other than text is not supported: there is no constrained decoding")

        include = body.get("include") or []

        for requested in include:
            if requested != _ENCRYPTED_REASONING:
                raise RequestError(f"include '{requested}' is not supported by this server")

        turns, names_by_id = [], {}

        if body.get("instructions"):
            turns.append({"role": "system", "content": body["instructions"]})

        items = body.get("input", "")
        items = [{"type": "message", "role": "user", "content": items}] if isinstance(items, str) else items
        reasoning = ""

        def open_assistant() -> dict:
            nonlocal reasoning

            if turns and turns[-1]["role"] == "assistant":
                turns[-1]["reasoning"] = turns[-1]["reasoning"] or reasoning
            else:
                turns.append(assistant_turn(reasoning=reasoning))

            reasoning = ""

            return turns[-1]

        for index, item in enumerate(items):
            kind = item.get("type", "message")
            where = f"input[{index}]"

            if kind == "message":
                role = item.get("role")
                content = text_of(item.get("content"), field_name=f"{where}.content")

                if role in ("system", "developer"):
                    turns.append({"role": "system", "content": content})
                elif role == "user":
                    turns.append({"role": "user", "content": content})
                elif role == "assistant":
                    turn = open_assistant()
                    turn["content"] = (turn["content"] + "\n" + content) if turn["content"] else content
                else:
                    raise RequestError(f"{where}.role '{role}' is not one this server reads")

            elif kind == "function_call":
                names_by_id[item.get("call_id", "")] = item.get("name", "")
                open_assistant()["tool_calls"].append({"name": item.get("name", ""), "arguments": item.get("arguments") or "{}"})

            elif kind == "function_call_output":
                output = item.get("output", "")
                turns.append(tool_turn(
                    names_by_id, item.get("call_id", ""),
                    output if isinstance(output, str) else text_of(output, field_name=f"{where}.output"),
                    field_name=f"{where}.call_id"))

            elif kind == "reasoning":
                reasoning = _reasoning_text(item)

            else:
                raise RequestError(f"{where} is a '{kind}' item, which this server does not read")

        variables = {}
        effort = (body.get("reasoning") or {}).get("effort")

        if effort is not None:
            variables.update(reasoning_effort_variables(protocol, effort, field_name="reasoning.effort"))

        tools = []

        if tool_choice_offers_tools(body.get("tool_choice")):
            for tool in body.get("tools") or []:
                if tool.get("type") != "function":
                    not_offered(tool.get("type", "unknown"), tool.get("name"))
                    continue

                declared = offered_tool(tool.get("name"), tool.get("description"), tool.get("parameters"))

                if declared is not None:
                    tools.append(declared)

        return ServedRequest(
            turns=turns,
            tools=tools,
            template_variables=variables,
            max_new_tokens=body.get("max_output_tokens"),
            temperature=body.get("temperature"),
            top_k=body.get("top_k"),
            top_p=body.get("top_p"),
            stream=bool(body.get("stream", False)),
            parallel_tool_calls=body.get("parallel_tool_calls", True) is not False,
            wire={"carry_reasoning": _ENCRYPTED_REASONING in include},
        )

    @staticmethod
    def _output(reply: ServedReply, context: ReplyContext) -> list[dict]:
        status, _ = _status(reply)
        output = []

        if reply.reasoning:
            item = {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex}",
                "summary": [{"type": "summary_text", "text": reply.reasoning}],
            }

            if context.wire.get("carry_reasoning"):
                item["encrypted_content"] = reply.reasoning

            output.append(item)

        if reply.content or not reply.tool_calls:
            output.append({
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": reply.content, "annotations": []}],
            })

        for call in reply.tool_calls:
            output.append({
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "status": "completed",
            })

        return output

    @staticmethod
    def _response(reply: ServedReply | None, context: ReplyContext, output: list[dict]) -> dict:
        status, incomplete = _status(reply) if reply is not None else ("in_progress", None)

        return {
            "id": context.response_id,
            "object": "response",
            "created_at": context.created,
            "model": context.model,
            "status": status,
            "incomplete_details": incomplete,
            "error": None,
            "output": output,
            "usage": None if reply is None else {
                "input_tokens": reply.prompt_tokens,
                "output_tokens": reply.completion_tokens,
                "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            },
        }

    def format(self, reply: ServedReply, context: ReplyContext) -> dict:
        return self._response(reply, context, self._output(reply, context))

    def stream_open(self, context: ReplyContext) -> str:
        return _event("response.created", {"response": self._response(None, context, [])})

    def stream_keepalive(self, context: ReplyContext) -> str:
        return _event("response.in_progress", {"response": self._response(None, context, [])})

    def stream_reply(self, reply: ServedReply, context: ReplyContext) -> str:
        output = self._output(reply, context)
        events = []

        for index, item in enumerate(output):
            item_id = item["id"]

            if item["type"] == "reasoning":
                text = item["summary"][0]["text"]
                part = {"item_id": item_id, "output_index": index, "summary_index": 0}
                events.append(_event("response.output_item.added", {"output_index": index, "item": {**item, "summary": []}}))
                events.append(_event("response.reasoning_summary_part.added", {**part, "part": {"type": "summary_text", "text": ""}}))
                events.append(_event("response.reasoning_summary_text.delta", {**part, "delta": text}))
                events.append(_event("response.reasoning_summary_text.done", {**part, "text": text}))
                events.append(_event("response.reasoning_summary_part.done", {**part, "part": {"type": "summary_text", "text": text}}))

            elif item["type"] == "message":
                text = item["content"][0]["text"]
                part = {"item_id": item_id, "output_index": index, "content_index": 0}
                events.append(_event("response.output_item.added", {"output_index": index, "item": {**item, "status": "in_progress", "content": []}}))
                events.append(_event("response.content_part.added", {**part, "part": {"type": "output_text", "text": "", "annotations": []}}))
                events.append(_event("response.output_text.delta", {**part, "delta": text}))
                events.append(_event("response.output_text.done", {**part, "text": text}))
                events.append(_event("response.content_part.done", {**part, "part": item["content"][0]}))

            else:
                call = {"item_id": item_id, "output_index": index}
                events.append(_event("response.output_item.added", {"output_index": index, "item": {**item, "arguments": "", "status": "in_progress"}}))
                events.append(_event("response.function_call_arguments.delta", {**call, "delta": item["arguments"]}))
                events.append(_event("response.function_call_arguments.done", {**call, "arguments": item["arguments"]}))

            events.append(_event("response.output_item.done", {"output_index": index, "item": item}))

        status, _ = _status(reply)
        events.append(_event(
            "response.completed" if status == "completed" else "response.incomplete",
            {"response": self._response(reply, context, output)}))

        return "".join(events)
