"""
What every endpoint reduces its request to, and the reply every endpoint formats (InferenceServerWire.md 3).

A request becomes the library's conversation -- turns in the binding's shape, the tools offered, the template
variables set -- and nothing family-specific: the grammar is the model's protocol, held by the worker. A reply
is the protocol's parse of what the model wrote, with the wire's bookkeeping added: call ids, why it ended.
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

#: Harness and UI tools a client runs itself and the model should never be asked to call. A server's policy
#: about its clients, not a property of any model.
_CLIENT_SIDE_TOOLS = {
    "update_plan", "request_user_input", "view_image",
    "spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent",
}


class RequestError(ValueError):
    """A request the server refuses, answered with a 400 that names what it refused."""


@dataclass
class ServedRequest:
    """One request, in the conversation's terms rather than any API's."""

    turns: list[dict]
    tools: list[dict] = field(default_factory=list)
    template_variables: dict[str, str] = field(default_factory=dict)
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    parallel_tool_calls: bool = True

    # What an endpoint's parse notes for its own formatter; nothing between the two reads it.
    wire: dict = field(default_factory=dict)

    @property
    def tools_json(self) -> str:
        return json.dumps(self.tools) if self.tools else ""


@dataclass
class ServedCall:
    call_id: str
    name: str
    arguments: str


@dataclass
class ServedReply:
    reasoning: str
    content: str
    tool_calls: list[ServedCall]
    finish_reason: str
    stop_sequence: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def call_id(name: str, arguments: str) -> str:
    """
    A correlation id derived from the call itself, so the same call has the same id each time a stateless
    request re-sends the transcript it is in.
    """
    digest = hashlib.sha256(f"{name}\x00{arguments}".encode()).hexdigest()[:24]

    return f"call_{digest}"


def offered_tool(name: str, description: str, parameters: dict | None) -> dict | None:
    """A tool in the OpenAI function shape every template reads, or None for one a client runs itself."""
    if not name or name in _CLIENT_SIDE_TOOLS:
        return None

    function = {"name": name, "description": description or ""}

    if parameters:
        function["parameters"] = parameters

    return {"type": "function", "function": function}


def not_offered(kind: str, name: str | None) -> None:
    """A tool a client listed that the server cannot run is never declared to the model, and the log says so."""
    _log.info("not offering %s tool %s: the server cannot run it", kind, name or "(unnamed)")


def text_of(content, *, field_name: str) -> str:
    """
    The text of a message's content, given as text or as a list of parts. A part the model cannot be given --
    an image, a file, audio -- is refused rather than dropped.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        raise RequestError(f"{field_name} must be text or a list of content parts")

    texts = []

    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue

        kind = part.get("type") if isinstance(part, dict) else None

        if kind in ("text", "input_text", "output_text"):
            texts.append(part.get("text", ""))
        else:
            raise RequestError(f"{field_name} holds a '{kind}' part, which this model cannot be given")

    return "".join(texts)


def assistant_turn(content: str = "", reasoning: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": [], "reasoning": reasoning}


def tool_turn(names_by_id: dict[str, str], identifier: str, content: str, *, field_name: str) -> dict:
    """A tool result as its own turn, named by the call it answers. An id that names no earlier call is refused."""
    name = names_by_id.get(identifier)

    if name is None:
        raise RequestError(f"{field_name} '{identifier}' answers no tool call earlier in the conversation")

    return {"role": "tool", "content": content, "tool_name": name}


def template_variable(protocol, name: str, value: str, *, field_name: str) -> tuple[str, str]:
    """
    A request field that sets one of the template's own variables, by exact value. A template without the
    variable, or a value it does not accept, is refused naming what is accepted -- never mapped onto another.
    """
    for variable in protocol.template_variables:
        if variable["name"] != name:
            continue

        if variable["values"] and value not in variable["values"]:
            raise RequestError(
                f"{field_name} '{value}' is not one this model's template accepts: {', '.join(variable['values'])}")

        return name, value

    raise RequestError(f"{field_name} is not a setting this model's template has")


def refuse_unless_default(body: dict, name: str, default) -> None:
    if name in body and body[name] is not None and body[name] != default:
        raise RequestError(f"{name} is not supported by this server")
