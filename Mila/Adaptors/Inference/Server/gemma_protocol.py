"""
Gemma 4 native chat + tool-call protocol for the Mila Inference Server.

Gemma 4 (the Mila checkpoint) uses registered vocabulary tokens for turns,
channels, and tool calls -- NOT a text convention. This module owns that
grammar so the OpenAI/Anthropic adapters can advertise tools, parse the model's
native <|tool_call> emission, and splice tool results back in, mirroring the
proven chat harness (Chat.GemmaToolCallParser / Chat.ChannelParser /
Chat.SystemPrompt). Grammar confirmed empirically against a live checkpoint:

  Turn:          <|turn>{role}\n{content}<turn|>
  Channel:       <|channel>{label}\n{reasoning}<channel|>{answer}
  Tool call:     <|tool_call>call:{name}{{key: "value", key2: 42}}<tool_call|>
  Tool response: <|tool_response>response:{name}{{key: "value"}}<tool_response|>

Namespaces on the call target (default_api:get_weather) are stripped to the
bare handler name.
"""
import hashlib
import json
import re
import uuid

# --- Template tokens (Mila Gemma checkpoint) ---
BOS = "<bos>"
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"
CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"
TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
TOOL_RESPONSE_OPEN = "<|tool_response>"
TOOL_RESPONSE_CLOSE = "<tool_response|>"
# Trained tool-DECLARATION tokens (Google Gemma 4 spec). Distinct from the
# tool-CALL tokens above: these wrap the schemas advertised in the system turn.
TOOL_DECL_OPEN = "<|tool>"
TOOL_DECL_CLOSE = "<tool|>"

# Gemma inconsistently wraps tool-call string values in this registered delimiter
# instead of plain double quotes (e.g. cmd: <|"|>ls -F<|"|>). Confirmed empirically
# against a live checkpoint -- the GemmaToolCallParser doc noted the vendor claim;
# it does occur. Parsed as an alternate string quote; stripped from text answers.
STRING_DELIM = '<|"|>'

# Thinking is off for agentic harnesses. Priming an empty thought channel
# suppresses "ghost" thought sections the 12B otherwise emits (mirrors the chat
# harness, Chat.ixx clearHistory).
THOUGHT_PRIME = f"{CHANNEL_OPEN}thought\n{CHANNEL_CLOSE}"

# Atomic special tokens stripped from decoded text before display. The tool-call
# and tool-response pairs are handled by the parser, not stripped blindly, but
# they are listed here so a residual marker never leaks into a text answer.
CONTROL_TOKENS = (
    BOS, "<eos>", "<pad>",
    TURN_OPEN, TURN_CLOSE,
    CHANNEL_OPEN, CHANNEL_CLOSE,
    "<|think|>",
    "<|tool>", "<tool|>", TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE,
    STRING_DELIM,
    "<end_of_turn>", "<start_of_turn>",
)

# Reasoning-channel labels (Chat.ChannelParser::isChannelLabel).
_CHANNEL_LABELS = ("thought", "thinking", "analysis", "reasoning")

# Harness/UI tools the model should never be asked to emit as a tool call.
_EXCLUDED_TOOLS = {
    "update_plan", "request_user_input", "view_image",
    "spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent",
}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def render_turn(role: str, content: str) -> str:
    """One closed Gemma turn: <|turn>{role}\\n{content}<turn|>\\n."""
    return f"{TURN_OPEN}{role}\n{content}{TURN_CLOSE}\n"


def assemble_prompt(system_block: str, turns: list[dict], continue_open: bool) -> str:
    """
    Assemble a full Gemma prompt from a system block and a role-tagged turn list.
    Each turn is {"role": "user"|"model", "content": str, "tool"?: bool}. When
    continue_open is set the final turn is emitted OPEN (no <turn|>) so generation
    resumes it -- used when the transcript ends on a tool exchange the model must
    continue. Shared by the OpenAI Responses and Anthropic Messages adapters so
    both drive the same native grammar.
    """
    parts: list[str] = [BOS]

    if system_block:
        parts.append(render_turn("system", system_block))

    body_turns = turns[:-1] if continue_open else turns
    for turn in body_turns:
        parts.append(render_turn(turn["role"], turn["content"]))

    if continue_open:
        last = turns[-1]
        # Leave the model turn open so generation resumes it after the tool result.
        parts.append(f"{TURN_OPEN}{last['role']}\n{last['content']}\n")
    else:
        parts.append(f"{TURN_OPEN}model\n")

    # Empty-thought prime: suppresses the ghost reasoning channels the 12B
    # otherwise emits (mirrors the chat harness). Load-bearing -- without it
    # generation degenerates.
    parts.append(THOUGHT_PRIME)
    return "".join(parts)


# Catch-all for pipe-bracketed registered tokens that are NOT in CONTROL_TOKENS
# (a sibling string delimiter or the bare <|> the checkpoint emits but this
# grammar has not enumerated). Left in place they leak verbatim into text
# answers and tool arguments. Two forms: the two-pipe delimiter family
# <|...|> (e.g. <|"|>) and the bare single-pipe <|>. Neither alternative
# matches the angle-form turn/channel markers (<|channel>, <|tool_call>),
# which are enumerated in CONTROL_TOKENS and stripped before this runs.
_PIPE_TOKEN = re.compile(r"<\|[^|>]*\|>|<\|>")


def strip_pipe_tokens(text: str) -> str:
    return _PIPE_TOKEN.sub("", text)


def strip_control_tokens(text: str) -> str:
    for token in CONTROL_TOKENS:
        if token in text:
            text = text.replace(token, "")

    # Scrub any residual <|...|> registered token the enumerated set missed.
    return strip_pipe_tokens(text)


# ---------------------------------------------------------------------------
# Tool advertisement (system prompt suffix)
# ---------------------------------------------------------------------------

def build_tool_injection(tools: list[dict], use_trained_declarations: bool = False) -> str:
    """
    Distill OpenAI function tool schemas into a system-turn suffix. Deliberately
    NO call-syntax instructions: Gemma emits tool CALLS via its trained
    <|tool_call> protocol, so teaching a foreign call format (e.g. Llama's
    <|python_tag|>) only confuses it (Chat.SystemPrompt.ixx).

    Two DECLARATION forms, selected by use_trained_declarations (A/B, see the MIS
    backlog). The default plain-text JSON list tells the model WHAT tools exist
    but gives it no anchor to the trained call grammar -- observed empirically to
    let the 12B improvise off-spec calls (call:bash:command=... with no
    <tool_call|> close) under the Claude Code harness. The trained form wraps each
    schema in the Google Gemma 4 <|tool>declaration:name{...}<tool|> token pair,
    on the theory that the trained declaration frame primes the trained call
    frame. Measure tool-selection reliability before making either the default.
    """
    normalized: list[dict] = []

    for tool in tools:
        if tool.get("type") not in (None, "function"):
            continue

        fn = tool.get("function") or tool
        name = fn.get("name")

        if not name or name in _EXCLUDED_TOOLS:
            continue

        normalized.append({
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })

    if not normalized:
        return ""

    if use_trained_declarations:
        return _build_trained_tool_declarations(normalized)

    return "\n\nYou have access to the following tools:\n" + json.dumps(normalized, indent=2)


def _build_trained_tool_declarations(normalized: list[dict]) -> str:
    """
    Render tool schemas in Gemma 4's trained declaration grammar:
    <|tool>declaration:name{description: <|"|>...<|"|>, parameters: {...json...}}<tool|>
    one per line in the system turn. String values use the trained <|"|> delimiter
    (mirrors _render_gemma_args); the parameters JSON-schema object is emitted as
    compact JSON.
    """
    declarations = []

    for tool in normalized:
        body = _render_gemma_args({
            "description": tool["description"],
            "parameters": tool["parameters"],
        })
        declarations.append(f"{TOOL_DECL_OPEN}declaration:{tool['name']}{{{body}}}{TOOL_DECL_CLOSE}")

    return "\n\n" + "\n".join(declarations)


# ---------------------------------------------------------------------------
# Tool-call parsing (model output -> Responses function_call)
# ---------------------------------------------------------------------------

def _strip_namespace(name: str) -> str:
    """Reduce default_api:get_weather / default_api.get_weather to get_weather."""
    for sep in (":", "."):
        idx = name.rfind(sep)
        if idx != -1:
            name = name[idx + 1:]

    return name.strip()


def _coerce_bare(bare: str):
    if bare == "true":
        return True
    if bare == "false":
        return False

    try:
        return int(bare)
    except ValueError:
        pass

    try:
        return float(bare)
    except ValueError:
        return bare


def _parse_arguments(args_body: str) -> dict:
    """
    Parse `key: "string", key2: 42` argument bodies. Mirrors
    GemmaToolCallParser::parseArguments: quoted strings terminate at the next
    unescaped quote; bare literals coerce to bool/number or fall back to string.
    """
    arguments: dict = {}
    remaining = args_body.strip()

    while remaining:
        colon = remaining.find(":")
        if colon == -1:
            break

        key = remaining[:colon].strip()
        remaining = remaining[colon + 1:].lstrip(" \t")

        if not remaining:
            break

        if remaining.startswith(STRING_DELIM):
            inner = len(STRING_DELIM)
            close = remaining.find(STRING_DELIM, inner)
            if close == -1:
                break
            arguments[key] = remaining[inner:close]
            remaining = remaining[close + len(STRING_DELIM):].lstrip(", \t")
        elif remaining[0] == '"':
            close = remaining.find('"', 1)
            if close == -1:
                break
            arguments[key] = remaining[1:close]
            remaining = remaining[close + 1:].lstrip(", \t")
        else:
            comma = remaining.find(",")
            if comma == -1:
                bare = remaining.strip()
                remaining = ""
            else:
                bare = remaining[:comma].strip()
                remaining = remaining[comma + 1:]
            arguments[key] = _coerce_bare(bare)

    return arguments


def _make_call_id(name: str, arguments: str) -> str:
    digest = hashlib.sha256(f"{name}\x00{arguments}".encode()).hexdigest()[:24]
    return f"call_{digest}"


def parse_tool_call(text: str) -> dict | None:
    """
    Detect and parse the most recent native Gemma tool call from accumulated
    model output. Returns a Responses API function_call item, or None if no
    <|tool_call> ... call:name{...} block is present.
    """
    open_pos = text.rfind(TOOL_CALL_OPEN)
    if open_pos == -1:
        return None

    body = text[open_pos + len(TOOL_CALL_OPEN):]

    call_pos = body.find("call:")
    if call_pos == -1:
        return None

    name_start = call_pos + len("call:")
    brace_open = body.find("{", name_start)
    if brace_open == -1:
        return None

    brace_close = body.rfind("}")
    if brace_close == -1 or brace_close <= brace_open:
        return None

    name = strip_pipe_tokens(_strip_namespace(body[name_start:brace_open].strip()))
    if not name:
        return None

    # Scrub any stray <|...|> token the checkpoint slipped into a string value
    # (a sibling delimiter this grammar has not enumerated). Left in place it
    # rides into the client's tool arguments (e.g. file_path "foo.cpp<|>").
    values = _parse_arguments(body[brace_open + 1:brace_close])
    values = {
        key: strip_pipe_tokens(value) if isinstance(value, str) else value
        for key, value in values.items()
    }
    arguments = json.dumps(values)

    return {
        "type": "function_call",
        "id": f"fc_{uuid.uuid4().hex}",
        "call_id": _make_call_id(name, arguments),
        "name": name,
        "arguments": arguments,
    }


# ---------------------------------------------------------------------------
# Tool-call / tool-response rendering (history replay -> prompt)
# ---------------------------------------------------------------------------

def _render_string_value(text: str) -> str:
    """
    Render a string value in Gemma's trained delimiter form: key:<|"|>value<|"|>.
    The value between the delimiter tokens is literal -- the trained format has no
    backslash escaping -- so guard only against an embedded delimiter that would
    otherwise close the span early (the delimiter analog of quote-escaping).
    """
    return f'{STRING_DELIM}{text.replace(STRING_DELIM, chr(34))}{STRING_DELIM}'


def _render_gemma_args(values: dict) -> str:
    parts = []

    for key, value in values.items():
        if isinstance(value, str):
            parts.append(f"{key}: {_render_string_value(value)}")
        else:
            parts.append(f"{key}: {json.dumps(value)}")

    return ", ".join(parts)


def format_tool_call(name: str, arguments_json: str) -> str:
    """Render an assistant tool call back into Gemma's native call grammar."""
    try:
        values = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        values = {}

    if not isinstance(values, dict):
        values = {}

    return f"{TOOL_CALL_OPEN}call:{name}{{{_render_gemma_args(values)}}}{TOOL_CALL_CLOSE}"


# When a client wraps tool output in a JSON envelope, the model-facing result is
# one of these fields; sibling fields (chunk ids, exit codes, timing) are metadata
# the model must NOT see, or it reports them as content.
_OUTPUT_KEYS = ("output", "result", "content", "stdout", "text")


def format_tool_response(name: str, result: str) -> str:
    """
    Render a client-executed tool result into Gemma's <|tool_response> grammar
    (GemmaToolCallParser::formatToolResponse). When the result is a JSON envelope
    only its primary output field is surfaced -- metadata siblings are dropped so
    the model does not echo chunk ids / exit codes as if they were content. A
    failed tool ({"content": "", "error": "..."}) has no usable output field, so
    its `error` is surfaced explicitly (it is NOT in _OUTPUT_KEYS); without it the
    model sees an empty result and blind-retries.
    """
    if not isinstance(result, str):
        try:
            result = json.dumps(result)
        except (TypeError, ValueError):
            result = str(result)

    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        # First NON-EMPTY output field -- an empty "content" must not shadow a
        # real "output"/"stdout", nor win over an "error" on a failed tool.
        output = next(
            (parsed[key] for key in _OUTPUT_KEYS
             if isinstance(parsed.get(key), str) and parsed[key].strip()),
            None,
        )
        error = parsed.get("error")
        has_error = isinstance(error, str) and error.strip()

        fields: dict = {}
        if output is not None:
            fields["result"] = output
        if has_error:
            fields["error"] = error
        if not fields:
            fields["result"] = json.dumps(parsed)

        body = _render_gemma_args(fields)
    else:
        body = f"result: {_render_string_value(result)}"

    return f"{TOOL_RESPONSE_OPEN}response:{name}{{{body}}}{TOOL_RESPONSE_CLOSE}"


# ---------------------------------------------------------------------------
# Answer extraction (model output -> user-facing text)
# ---------------------------------------------------------------------------

def _remove_spans(text: str, open_token: str, close_token: str) -> str:
    while True:
        start = text.find(open_token)
        if start == -1:
            return text

        end = text.find(close_token, start + len(open_token))
        if end == -1:
            return text[:start]

        text = text[:start] + text[end + len(close_token):]


def _strip_tool_spans(text: str, open_token: str, close_token: str) -> str:
    """
    Remove tool-call / tool-response spans for the answer-text path. A COMPLETE
    open..close span is dropped whole. A DANGLING open (no close) has ONLY its
    marker removed, keeping the body -- so a malformed or truncated tool call
    (e.g. the off-spec call:name:key=value form the 12B emits with no
    <tool_call|> close) degrades to readable text instead of blanking the turn.

    This is the safety net for the case parse_tool_call could not classify:
    _remove_spans would truncate from the first open, and a response that STARTS
    with an unclosed <|tool_call> would collapse to an empty string. Residual
    open markers are cleared afterward by strip_control_tokens.
    """
    result = []
    cursor = 0

    while True:
        start = text.find(open_token, cursor)
        if start == -1:
            result.append(text[cursor:])
            break

        end = text.find(close_token, start + len(open_token))
        result.append(text[cursor:start])

        if end == -1:
            # Dangling open: skip only the marker, keep the body that follows.
            cursor = start + len(open_token)
        else:
            cursor = end + len(close_token)

    return "".join(result)


def extract_answer(text: str) -> str:
    """
    Reduce a channel-structured Gemma response to just the user-facing answer:
    drop every tool-call/response span and every reasoning-channel span
    (<|channel>label\\nreasoning<channel|>), then strip residual control tokens.

    Reasoning channels can appear more than once and interleaved with answer
    text -- the 12B emits mid-answer thought channels on the agentic path
    DESPITE the empty-thought prime -- so ALL channel spans are removed, not
    just a leading run. The previous single-pass logic left interior/trailing
    channels intact, and strip_control_tokens erased only the <|channel> markers
    while leaving the label + reasoning body as literal text (a "thought\\n..."
    block leaking into the answer, which is also why that content never became a
    tool call). _remove_spans drops an unclosed trailing channel too: a response
    cut off mid-reasoning yields the answer prefix, not a leaked reasoning tail.
    """
    text = _strip_tool_spans(text, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
    text = _strip_tool_spans(text, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE)
    text = _remove_spans(text, CHANNEL_OPEN, CHANNEL_CLOSE)

    return strip_control_tokens(text).strip()
