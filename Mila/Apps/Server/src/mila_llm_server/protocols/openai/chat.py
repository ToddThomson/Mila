"""
OpenAI Chat Completions and Completions.
  Chat:        POST /v1/chat/completions
  Completions: POST /v1/completions

Chat Completions is a parser into the conversation and a formatter from the reply, with nothing about which
model is loaded (InferenceServerWire.md 3.3); what it honours and refuses is 3.6's table.
"""
import json
import time
import uuid

from mila_llm_server.config import loaded, settings
from mila_llm_server.conversation import (
    RequestError,
    ServedReply,
    ServedRequest,
    assistant_turn,
    not_offered,
    offered_tool,
    refuse_unless_default,
    template_variable,
    text_of,
    tool_turn,
)
from mila_llm_server.protocols.base import ChatEndpoint, CompletionsCapable, ReplyContext
from mila_llm_server.protocols.utils import (
    END_TURN,
    openai_finish_reason,
    parse_completion_prompt,
    parse_stop,
)
from mila_llm_server.schemas.internal import InferenceRequest, InferenceResponse


def tool_choice_offers_tools(tool_choice, *, field_name: str = "tool_choice") -> bool:
    """`auto` (or absent) offers the tools and `none` does not. Forcing a call is refused: asking is not doing."""
    if tool_choice in (None, "auto"):
        return True

    if tool_choice == "none":
        return False

    raise RequestError(f"{field_name} {json.dumps(tool_choice)} is not supported: the model cannot be made to call a tool")


def _offered_tools(tools: list | None) -> list[dict]:
    offered = []

    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            not_offered(str(tool.get("type") if isinstance(tool, dict) else tool), None)
            continue

        function = tool.get("function") or {}
        declared = offered_tool(function.get("name"), function.get("description"), function.get("parameters"))

        if declared is not None:
            offered.append(declared)

    return offered


class OpenAIChatCompletions(ChatEndpoint):

    @property
    def path(self) -> str:
        return "/v1/chat/completions"

    def parse(self, body: dict, protocol) -> ServedRequest:
        if body.get("n", 1) != 1:
            raise RequestError("n must be 1: one completion is served per request")

        if body.get("logprobs") or body.get("top_logprobs"):
            raise RequestError("logprobs are not supported by this server")

        if body.get("seed") is not None:
            raise RequestError("seed is not supported by this server")

        for name in ("frequency_penalty", "presence_penalty"):
            refuse_unless_default(body, name, 0)

        if body.get("logit_bias"):
            raise RequestError("logit_bias is not supported by this server")

        response_format = body.get("response_format")

        if response_format is not None and response_format.get("type", "text") != "text":
            raise RequestError("response_format other than text is not supported: there is no constrained decoding")

        turns, names_by_id = [], {}

        for index, message in enumerate(body.get("messages") or []):
            role = message.get("role")
            where = f"messages[{index}]"

            if role in ("system", "developer"):
                turns.append({"role": "system", "content": text_of(message.get("content"), field_name=f"{where}.content")})

            elif role == "user":
                turns.append({"role": "user", "content": text_of(message.get("content"), field_name=f"{where}.content")})

            elif role == "assistant":
                turn = assistant_turn(
                    text_of(message.get("content"), field_name=f"{where}.content"),
                    message.get("reasoning_content") or "")

                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    names_by_id[call.get("id", "")] = function.get("name", "")
                    turn["tool_calls"].append({"name": function.get("name", ""), "arguments": function.get("arguments") or "{}"})

                turns.append(turn)

            elif role == "tool":
                turns.append(tool_turn(
                    names_by_id, message.get("tool_call_id", ""),
                    text_of(message.get("content"), field_name=f"{where}.content"), field_name=f"{where}.tool_call_id"))

            else:
                raise RequestError(f"{where}.role '{role}' is not one this server reads")

        variables = {}

        if body.get("reasoning_effort") is not None:
            name, value = template_variable(protocol, "reasoning_effort", body["reasoning_effort"], field_name="reasoning_effort")
            variables[name] = value

        stream_options = body.get("stream_options") or {}

        return ServedRequest(
            turns=turns,
            tools=_offered_tools(body.get("tools")) if tool_choice_offers_tools(body.get("tool_choice")) else [],
            template_variables=variables,
            max_new_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
            temperature=body.get("temperature"),
            top_k=body.get("top_k"),
            top_p=body.get("top_p"),
            stop=parse_stop(body.get("stop")),
            stream=bool(body.get("stream", False)),
            parallel_tool_calls=body.get("parallel_tool_calls", True) is not False,
            wire={"include_usage": bool(stream_options.get("include_usage", False))},
        )

    @staticmethod
    def _message(reply: ServedReply) -> dict:
        message = {"role": "assistant", "content": reply.content if reply.content or not reply.tool_calls else None}

        if reply.reasoning:
            message["reasoning_content"] = reply.reasoning

        if reply.tool_calls:
            message["tool_calls"] = [
                {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in reply.tool_calls]

        return message

    @staticmethod
    def _usage(reply: ServedReply) -> dict:
        return {
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
        }

    def format(self, reply: ServedReply, context: ReplyContext) -> dict:
        return {
            "id": context.response_id,
            "object": "chat.completion",
            "created": context.created,
            "model": context.model,
            "choices": [{"index": 0, "message": self._message(reply), "finish_reason": openai_finish_reason(reply.finish_reason)}],
            "usage": self._usage(reply),
        }

    @staticmethod
    def _chunk(context: ReplyContext, delta: dict, finish_reason: str | None = None) -> str:
        chunk = {
            "id": context.response_id,
            "object": "chat.completion.chunk",
            "created": context.created,
            "model": context.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

        return f"data: {json.dumps(chunk)}\n\n"

    def stream_open(self, context: ReplyContext) -> str:
        return self._chunk(context, {"role": "assistant", "content": ""})

    def stream_keepalive(self, context: ReplyContext) -> str:
        return ": keepalive\n\n"

    def stream_reply(self, reply: ServedReply, context: ReplyContext) -> str:
        events = []

        if reply.reasoning:
            events.append(self._chunk(context, {"reasoning_content": reply.reasoning}))

        if reply.content:
            events.append(self._chunk(context, {"content": reply.content}))

        for index, call in enumerate(reply.tool_calls):
            events.append(self._chunk(context, {"tool_calls": [{
                "index": index, "id": call.call_id, "type": "function",
                "function": {"name": call.name, "arguments": call.arguments}}]}))

        events.append(self._chunk(context, {}, openai_finish_reason(reply.finish_reason)))

        if context.wire.get("include_usage"):
            usage = {
                "id": context.response_id, "object": "chat.completion.chunk", "created": context.created,
                "model": context.model, "choices": [], "usage": self._usage(reply)}
            events.append(f"data: {json.dumps(usage)}\n\n")

        events.append("data: [DONE]\n\n")

        return "".join(events)

    def error(self, message: str) -> dict:
        return {"error": {"type": "invalid_request_error", "message": message}}


class OpenAICompletions(CompletionsCapable):
    """Raw continuation of a prompt the caller rendered or tokenized itself."""

    @property
    def completions_path(self) -> str:
        return "/v1/completions"

    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]:
        prompt_str, prompt_ids = parse_completion_prompt(body.get("prompt", ""))
        req = InferenceRequest(
            prompt_ids=prompt_ids,
            max_new_tokens=body.get("max_tokens", settings.default_max_new_tokens),
            temperature=body.get("temperature", settings.default_temperature),
            top_k=body.get("top_k", settings.default_top_k),
            top_p=body.get("top_p", settings.default_top_p),
            stream=body.get("stream", False),
            stop=parse_stop(body.get("stop")),
        )
        return prompt_str, req

    def format_completions_response(self, response: InferenceResponse) -> dict:
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": loaded.name,
            "choices": [{"text": response.text, "index": 0, "finish_reason": openai_finish_reason(response.finish_reason)}],
            "usage": {
                "prompt_tokens": response.prompt_token_count,
                "completion_tokens": response.completion_token_count,
                "total_tokens": response.prompt_token_count + response.completion_token_count,
            },
        }

    def format_stream_chunk(self, text: str, done: bool, finish_reason: str = END_TURN) -> str:
        chunk = {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": loaded.name,
            "choices": [{"text": text, "index": 0, "finish_reason": openai_finish_reason(finish_reason) if done else None}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_stream_done(self) -> str:
        return "data: [DONE]\n\n"
