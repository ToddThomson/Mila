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


def strip_control_tokens(text: str) -> str:
    for token in CONTROL_TOKENS:
        if token in text:
            text = text.replace(token, "")

    return text


# ---------------------------------------------------------------------------
# Tool advertisement (system prompt suffix)
# ---------------------------------------------------------------------------

def build_tool_injection(tools: list[dict]) -> str:
    """
    Distill OpenAI function tool schemas into a plain description appended to the
    system turn. Deliberately NO call-syntax instructions: Gemma emits tool calls
    via its trained <|tool_call> protocol, so teaching a foreign format (e.g.
    Llama's <|python_tag|>) only confuses it (Chat.SystemPrompt.ixx).
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

    return "\n\nYou have access to the following tools:\n" + json.dumps(normalized, indent=2)


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

    name = _strip_namespace(body[name_start:brace_open].strip())
    if not name:
        return None

    arguments = json.dumps(_parse_arguments(body[brace_open + 1:brace_close]))

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

def _render_gemma_args(values: dict) -> str:
    parts = []

    for key, value in values.items():
        if isinstance(value, str):
            parts.append(f'{key}: "{value}"')
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


def _escape_value(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def format_tool_response(name: str, result: str) -> str:
    """
    Render a client-executed tool result into Gemma's <|tool_response> grammar
    (GemmaToolCallParser::formatToolResponse). When the result is a JSON envelope
    only its primary output field is surfaced -- metadata siblings are dropped so
    the model does not echo chunk ids / exit codes as if they were content.
    """
    if not isinstance(result, str):
        try:
            result = json.dumps(result)
        except (TypeError, ValueError):
            result = str(result)

    body_text = result

    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        body_text = next(
            (parsed[key] for key in _OUTPUT_KEYS if isinstance(parsed.get(key), str)),
            None,
        )
        if body_text is None:
            body_text = json.dumps(parsed)

    return f'{TOOL_RESPONSE_OPEN}response:{name}{{result: "{_escape_value(body_text)}"}}{TOOL_RESPONSE_CLOSE}'


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


def extract_answer(text: str) -> str:
    """
    Reduce a channel-structured Gemma response to just the user-facing answer:
    drop any tool-call/response spans, take the text after the final reasoning
    channel (Chat.ChannelParser), and strip residual control tokens.
    """
    text = _remove_spans(text, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
    text = _remove_spans(text, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE)

    first_open = text.find(CHANNEL_OPEN)

    if first_open != -1:
        prefix = text[:first_open]
        cursor = first_open

        while True:
            header_start = cursor + len(CHANNEL_OPEN)
            close = text.find(CHANNEL_CLOSE, header_start)

            if close == -1:
                # Stopped inside the reasoning channel: no answer emitted.
                text = prefix
                break

            after_close = close + len(CHANNEL_CLOSE)
            next_open = text.find(CHANNEL_OPEN, after_close)
            between = text[after_close:next_open] if next_open != -1 else text[after_close:]

            if next_open != -1 and between.strip() == "":
                cursor = next_open
                continue

            text = prefix + text[after_close:]
            break

    return strip_control_tokens(text).strip()
