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
    presence_penalty: float | None = None
    seed: int | None = None
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


def placed_system_turns(turns: list[dict], renders_later_system_turns: bool = True) -> list[dict]:
    """
    The conversation with its system text where a template can take it (InferenceServerWire.md 3.6). Every system
    turn before the model's first reply joins one leading turn, in the order sent: each reply stands after all of
    them, so the joined turn is what every reply was told. A later one stays in place where the template renders
    it, and is folded into the conversation where it does not; one with no text is dropped, as it renders nothing.
    """
    first_reply = next((index for index, turn in enumerate(turns) if turn["role"] == "assistant"), len(turns))
    opening = turns[:first_reply]
    system_texts = [turn["content"] for turn in opening if turn["role"] == "system" and turn["content"].strip()]
    placed = [{"role": "system", "content": "\n\n".join(system_texts)}] if system_texts else []
    placed += [turn for turn in opening if turn["role"] != "system"]
    later = [turn for turn in turns[first_reply:] if turn["role"] != "system" or turn["content"].strip()]

    return placed + (later if renders_later_system_turns else _folded_system_turns(later))


def _folded_system_turns(turns: list[dict]) -> list[dict]:
    """
    Each system turn's text joined to the end of the user message or tool result before it, or, after a reply, to
    the start of the next one. It keeps its place after the replies that preceded it, and no turn is added: a new
    user turn would move the template's last query and change how the replies before it render. One with no turn
    to join stays in place, for the template to refuse.
    """
    folded: list[dict] = []
    waiting: list[str] = []

    for turn in turns:
        if turn["role"] == "system":
            if folded and folded[-1]["role"] in ("user", "tool"):
                folded[-1] = dict(folded[-1], content=f"{folded[-1]['content']}\n\n{turn['content']}")
            else:
                waiting.append(turn["content"])

            continue

        if waiting and turn["role"] in ("user", "tool"):
            turn = dict(turn, content="\n\n".join([*waiting, turn["content"]]))
            waiting = []

        folded.append(turn)

    return folded + [{"role": "system", "content": text} for text in waiting]


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


def has_template_variable(protocol, name: str) -> bool:
    return any(variable["name"] == name for variable in protocol.template_variables)


def chat_template_kwargs_variables(protocol, kwargs, *, field_name: str = "chat_template_kwargs") -> dict[str, str]:
    """
    vLLM's field for a template's own variables, which other local servers take too (InferenceServerWire.md 3.6).
    Each key is one of the template's variables and each value one it accepts, given as text. `enable_thinking`
    false is met as sent by a template without the switch, as `reasoning_effort` `none` is. Any other key the
    template does not have is refused naming the ones it has: vLLM drops it, which leaves the client believing it
    was set.
    """
    if kwargs is None:
        return {}

    if not isinstance(kwargs, dict):
        raise RequestError(f"{field_name} must be an object")

    variables = {}

    for key, given in kwargs.items():
        where = f"{field_name}.{key}"
        value = _template_text(given, field_name=where)

        if not has_template_variable(protocol, key):
            if key == "enable_thinking" and value == "false":
                continue

            names = ", ".join(variable["name"] for variable in protocol.template_variables) or "none"
            raise RequestError(f"{where} is not a setting this model's template has. Its settings: {names}")

        name, value = template_variable(protocol, key, value, field_name=where)
        variables[name] = value

    return variables


def _template_text(value, *, field_name: str) -> str:
    """A JSON value as the template variable's text: JSON's true and false are the templates' own 'true' and 'false'."""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return value

    if isinstance(value, int):
        return str(value)

    raise RequestError(f"{field_name} must be true, false, text or an integer")


def with_effort_variables(variables: dict[str, str], effort: str, effort_variables: dict[str, str], *,
                          field_name: str) -> dict[str, str]:
    """
    The template variables a request set directly, joined with those its reasoning effort sets. Where the two
    disagree the request is refused naming both, never settled by one of them winning.
    """
    for name, value in effort_variables.items():
        if variables.get(name, value) != value:
            raise RequestError(
                f"{field_name} '{effort}' sets {name} to '{value}', which contradicts chat_template_kwargs.{name} '{variables[name]}'")

    if effort != "none" and variables.get("enable_thinking") == "false":
        raise RequestError(
            f"{field_name} '{effort}' asks for reasoning, which contradicts chat_template_kwargs.enable_thinking 'false'")

    return {**variables, **effort_variables}


def reasoning_effort_variables(protocol, effort: str, *, field_name: str) -> dict[str, str]:
    """
    OpenAI's reasoning switch onto the template's own variables (InferenceServerWire.md 3.6): `none` is thinking
    off, a level is the template's own level by exact name, or thinking on where the template has no levels.
    """
    if effort == "none":
        if not has_template_variable(protocol, "enable_thinking"):
            return {}

        name, value = template_variable(protocol, "enable_thinking", "false", field_name=field_name)

        return {name: value}

    if has_template_variable(protocol, "reasoning_effort"):
        name, value = template_variable(protocol, "reasoning_effort", effort, field_name=field_name)

        return {name: value}

    if has_template_variable(protocol, "enable_thinking"):
        name, value = template_variable(protocol, "enable_thinking", "true", field_name=field_name)

        return {name: value}

    raise RequestError(f"{field_name} '{effort}' is not supported: this model does not reason. Send 'none' or omit it")


def with_thinking_by_default(protocol, variables: dict[str, str]) -> dict[str, str]:
    """
    The template variables with thinking on where the request said nothing about it (InferenceServerWire.md 10.6).
    The server's clients are agents, so a request naming neither thinking nor an effort gets the model's thinking
    at the template's own default level. A request that names either is left as it asked, and a template with no
    thinking switch is unchanged.
    """
    if "enable_thinking" in variables or "reasoning_effort" in variables:
        return variables

    if not has_template_variable(protocol, "enable_thinking"):
        return variables

    name, value = template_variable(protocol, "enable_thinking", "true", field_name="enable_thinking")

    return {**variables, name: value}


def presence_penalty_of(body: dict) -> float | None:
    """OpenAI's presence penalty, in OpenAI's own range; absent is None, the model's recommended setting."""
    value = body.get("presence_penalty")

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -2.0 <= value <= 2.0:
        raise RequestError("presence_penalty must be a number from -2.0 to 2.0")

    return float(value)


def seed_of(body: dict) -> int | None:
    """
    OpenAI's sampling seed, a 64-bit signed integer, as the binding takes it: the same 64 bits read unsigned. Absent
    is None, and sampling continues from the previous reply.
    """
    value = body.get("seed")

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int) or not -2**63 <= value < 2**63:
        raise RequestError("seed must be a 64-bit integer")

    return value % 2**64


def refuse_unless_default(body: dict, name: str, default) -> None:
    if name in body and body[name] is not None and body[name] != default:
        raise RequestError(f"{name} is not supported by this server")
