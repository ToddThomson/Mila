"""
Gemma 4 native chat + tool-call protocol for the Mila Inference Server.

Gemma 4 (the Mila checkpoint) uses registered vocabulary tokens for turns,
channels, and tool calls -- NOT a text convention. This module owns that
grammar so the OpenAI/Anthropic adapters can advertise tools, parse the model's
native <|tool_call> emission, and splice tool results back in, mirroring the
proven chat harness (Chat.GemmaToolCallParser / Chat.ChannelParser /
Chat.SystemPrompt). Grammar confirmed empirically against a live checkpoint and
reconciled 2026-07-16 against Google's canonical chat_template.jinja
(https://huggingface.co/google/gemma-4-12B-it/raw/main/chat_template.jinja), which
is the reference spec -- MIS builds prompts natively rather than running Jinja:

  Turn:          <|turn>{role}\n{content}<turn|>
  Channel:       <|channel>{label}\n{reasoning}<channel|>{answer}
  Declaration:   <|tool>declaration:{name}{{description:<|"|>..<|"|>,parameters:{{..}}}}<tool|>
  Tool call:     <|tool_call>call:{name}{{key:<|"|>value<|"|>,key2:42}}<tool_call|>
  Tool response: <|tool_response>response:{name}{{key:<|"|>value<|"|>}}<tool_response|>

Rendering rules that are NOT cosmetic: no whitespace around ':' or ',', keys in
sorted order, string values in the <|"|> delimiter, and containers recursed into
the same grammar rather than emitted as JSON. This is a tokenized format -- "key:
value" and "key:value" are different tokens.

Namespaces on the call target (default_api:get_weather) are stripped to the
bare handler name.

Render and parse are inverses over the value grammar (recursive descent both ways),
pinned by the round-trip oracle in tests/test_gemma_protocol.py. The renderer is
STRICT (emits only the trained form); the parser is LENIENT (also accepts plain
quotes and stray whitespace), because it reads what the model emits and the model
is inconsistent.
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
        # Resume the model's OPEN turn: emit its content so far and stop dead, so the
        # next token continues it. NO thought prime and NO trailing newline here --
        # the reference template emits nothing after a tool response when thinking is
        # off, and this turn already carries its thought channel from before the call.
        # Priming a SECOND, empty one mid-turn is off-distribution: the model parrots
        # it back (measured: empty-channel echoes, one 682-char runaway loop).
        parts.append(f"{TURN_OPEN}{last['role']}\n{last['content']}")
    else:
        # Opening a fresh model turn. The empty-thought prime belongs HERE and only
        # here -- it suppresses the ghost reasoning channels the 12B otherwise emits
        # (load-bearing: without it generation degenerates), and it matches both the
        # reference template and the doc's turn-start remediation form.
        parts.append(f"{TURN_OPEN}model\n")
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

def build_tool_injection(tools: list[dict]) -> str:
    """
    Distill OpenAI function tool schemas into a system-turn suffix, in Gemma 4's
    trained <|tool>declaration:name{...}<tool|> grammar.

    Deliberately NO call-syntax instructions: Gemma emits tool CALLS via its trained
    <|tool_call> protocol, so teaching a foreign call format (e.g. Llama's
    <|python_tag|>) only confuses it (Chat.SystemPrompt.ixx).

    There is ONE declaration form because Gemma has one. A plain-text JSON list used
    to sit behind a `use_trained_declarations` A/B toggle here; it was removed once
    Google published the canonical template, since it was never a Gemma format at all
    -- it was prose plus a JSON dump, invented before the format was known, and the
    reference-parity test can never match it. The A/B measured it losing anyway (the
    12B invented tools that do not exist when given the JSON list); that evidence is
    recorded in the BACKLOG, which is where a settled question belongs rather than a
    live switch.
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

    return _build_trained_tool_declarations(normalized)


def _render_required(required) -> str:
    """A required-list body: <|"|>a<|"|>,<|"|>b<|"|> (the names are delimiter-wrapped)."""
    return ",".join(_render_string_value(str(item)) for item in required)


def _render_declaration_items(items: dict) -> str:
    """
    The items:{...} body of an ARRAY property. Mirrors the template's items branch,
    including its quirks: `type` is uppercased and delimiter-wrapped, and any other
    free-form key renders through format_argument with escape_keys defaulted TRUE
    (so nested mapping keys there ARE delimiter-wrapped, unlike in argument bodies).
    """
    parts = []

    for key, value in sorted(items.items(), key=lambda entry: entry[0]):
        if value is None:
            continue

        if key == "properties":
            body = _render_parameters(value) if isinstance(value, dict) else ""
            parts.append(f"properties:{{{body}}}")
        elif key == "required":
            parts.append(f"required:[{_render_required(value)}]")
        elif key == "type":
            if isinstance(value, str):
                parts.append(f"type:{_render_string_value(value.upper())}")
            else:
                parts.append(f"type:{_render_value([str(item).upper() for item in value])}")
        else:
            parts.append(f"{key}:{_render_value(value, escape_keys=True)}")

    return ",".join(parts)


def _render_property(schema: dict) -> str:
    """
    One JSON-schema property in the trained declaration grammar, e.g.
    {description:<|"|>The city<|"|>,type:<|"|>STRING<|"|>}.

    Field order is POSITIONAL, not alphabetical (description, enum/items, nullable,
    properties/required, type) -- that is the order the template emits and the order
    the model saw in training, so it is not ours to tidy. Types are UPPERCASED.
    """
    parts = []
    description = schema.get("description")

    if description:
        parts.append(f"description:{_render_string_value(description)}")

    schema_type = str(schema.get("type", "")).upper()

    if schema_type == "STRING" and schema.get("enum"):
        parts.append(f"enum:{_render_value(schema['enum'], escape_keys=True)}")
    elif schema_type == "ARRAY" and isinstance(schema.get("items"), dict) and schema["items"]:
        parts.append(f"items:{{{_render_declaration_items(schema['items'])}}}")

    if schema.get("nullable"):
        parts.append("nullable:true")

    if schema_type == "OBJECT":
        properties = schema.get("properties")

        if isinstance(properties, dict):
            parts.append(f"properties:{{{_render_parameters(properties)}}}")

        if schema.get("required"):
            parts.append(f"required:[{_render_required(schema['required'])}]")

    parts.append(f"type:{_render_string_value(schema_type)}")

    return "{" + ",".join(parts) + "}"


def _render_parameters(properties: dict) -> str:
    """The properties:{...} body: name:{...},name2:{...} with names in sorted order."""
    parts = []

    for name, schema in sorted(properties.items(), key=lambda entry: entry[0]):
        if isinstance(schema, dict):
            parts.append(f"{name}:{_render_property(schema)}")

    return ",".join(parts)


def _build_trained_tool_declarations(normalized: list[dict]) -> str:
    """
    Render tool schemas in Gemma 4's trained declaration grammar, mirroring
    format_function_declaration in Google's canonical chat_template.jinja:

      <|tool>declaration:name{description:<|"|>..<|"|>,parameters:{properties:{
      city:{description:<|"|>..<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>city<|"|>],
      type:<|"|>OBJECT<|"|>}}<tool|>

    The parameters schema is a full DSL, NOT the compact JSON blob this used to emit.
    Declarations are concatenated with no separator and no leading blank line, which is
    how the template appends them to the system turn -- <|tool> is an atomic special
    token, so it needs no whitespace to separate it from the system prose.

    ONE deliberate deviation from the template: it closes the parameters block from
    inside its `type:` branch, so a schema with no `type` leaves the block unclosed and
    a comma dangling. We always close it. Malformed either way, but ours stays parseable.
    """
    declarations = []

    for tool in normalized:
        parts = [f"description:{_render_string_value(tool['description'])}"]
        params = tool.get("parameters")

        if params:
            param_parts = []
            properties = params.get("properties")

            if properties:
                param_parts.append(f"properties:{{{_render_parameters(properties)}}}")

            if params.get("required"):
                param_parts.append(f"required:[{_render_required(params['required'])}]")

            if params.get("type"):
                param_parts.append(f"type:{_render_string_value(str(params['type']).upper())}")

            parts.append("parameters:{" + ",".join(param_parts) + "}")

        body = ",".join(parts)
        declarations.append(f"{TOOL_DECL_OPEN}declaration:{tool['name']}{{{body}}}{TOOL_DECL_CLOSE}")

    return "".join(declarations)


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
    if bare == "null":
        return None
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


class _Cursor:
    """
    A read cursor over an argument body, for the recursive-descent parse.

    The grammar nests (objects and arrays hold values that are themselves objects and
    arrays), so a single left-to-right scan with a shared position is the only thing
    that composes. The predecessor searched for the next ',' from the start of each
    value and so could not see container boundaries at all -- it truncated the value
    AND spilled the container's remaining contents out as sibling arguments.
    """
    __slots__ = ("text", "position")

    def __init__(self, text: str):
        self.text = text
        self.position = 0

    def at_end(self) -> bool:
        return self.position >= len(self.text)

    def peek(self) -> str:
        return self.text[self.position]

    def starts_with(self, token: str) -> bool:
        return self.text.startswith(token, self.position)

    def skip_whitespace(self) -> None:
        while self.position < len(self.text) and self.text[self.position] in " \t\r\n":
            self.position += 1


def _parse_delimited_string(cursor: _Cursor) -> str:
    """Consume a <|"|> span. Literal -- the trained format has no backslash escaping."""
    cursor.position += len(STRING_DELIM)
    close = cursor.text.find(STRING_DELIM, cursor.position)

    if close == -1:
        # Unterminated: take the rest rather than discarding the argument.
        value = cursor.text[cursor.position:]
        cursor.position = len(cursor.text)
        return value

    value = cursor.text[cursor.position:close]
    cursor.position = close + len(STRING_DELIM)
    return value


def _parse_quoted_string(cursor: _Cursor) -> str:
    """Consume a plain "..." span -- the alternate quoting the model also emits."""
    cursor.position += 1
    close = cursor.text.find('"', cursor.position)

    if close == -1:
        value = cursor.text[cursor.position:]
        cursor.position = len(cursor.text)
        return value

    value = cursor.text[cursor.position:close]
    cursor.position = close + 1
    return value


def _parse_bare(cursor: _Cursor):
    """Consume a bare literal, stopping at whatever closes the enclosing construct."""
    start = cursor.position

    while not cursor.at_end() and cursor.peek() not in ",}]":
        cursor.position += 1

    return _coerce_bare(cursor.text[start:cursor.position].strip())


def _parse_key(cursor: _Cursor) -> str:
    """
    Consume a key: bare, or delimiter-/quote-wrapped (declarations wrap keys, and the
    model sometimes mirrors that inside call arguments).
    """
    cursor.skip_whitespace()

    if cursor.starts_with(STRING_DELIM):
        return _parse_delimited_string(cursor)

    if not cursor.at_end() and cursor.peek() == '"':
        return _parse_quoted_string(cursor)

    start = cursor.position

    while not cursor.at_end() and cursor.peek() not in ":,}":
        cursor.position += 1

    return cursor.text[start:cursor.position].strip()


def _parse_object(cursor: _Cursor) -> dict:
    result: dict = {}
    cursor.position += 1
    cursor.skip_whitespace()

    if not cursor.at_end() and cursor.peek() == "}":
        cursor.position += 1
        return result

    while not cursor.at_end():
        key = _parse_key(cursor)
        cursor.skip_whitespace()

        if cursor.at_end() or cursor.peek() != ":":
            break

        cursor.position += 1
        result[key] = _parse_value(cursor)
        cursor.skip_whitespace()

        if not cursor.at_end() and cursor.peek() == ",":
            cursor.position += 1
            cursor.skip_whitespace()
            continue

        if not cursor.at_end() and cursor.peek() == "}":
            cursor.position += 1

        break

    return result


def _parse_array(cursor: _Cursor) -> list:
    result: list = []
    cursor.position += 1
    cursor.skip_whitespace()

    if not cursor.at_end() and cursor.peek() == "]":
        cursor.position += 1
        return result

    while not cursor.at_end():
        result.append(_parse_value(cursor))
        cursor.skip_whitespace()

        if not cursor.at_end() and cursor.peek() == ",":
            cursor.position += 1
            continue

        if not cursor.at_end() and cursor.peek() == "]":
            cursor.position += 1

        break

    return result


def _parse_value(cursor: _Cursor):
    """Parse one value of the trained grammar -- the inverse of _render_value."""
    cursor.skip_whitespace()

    if cursor.at_end():
        return None

    if cursor.starts_with(STRING_DELIM):
        return _parse_delimited_string(cursor)

    if cursor.peek() == '"':
        return _parse_quoted_string(cursor)

    if cursor.peek() == "{":
        return _parse_object(cursor)

    if cursor.peek() == "[":
        return _parse_array(cursor)

    return _parse_bare(cursor)


def _parse_arguments(args_body: str) -> dict:
    """
    Parse a `key:value,key2:42` argument body into a dict.

    The inverse of _render_gemma_args, and recursive: container values parse back to
    containers. Whitespace around ':' and ',' is tolerated on the way in even though
    the trained format emits none.

    Malformed input degrades rather than raising -- a truncated body keeps the
    arguments parsed so far, so a cut-off tool call surfaces partially instead of
    blanking. It is NOT strict: this reads what the model emits, and the model is
    inconsistent (see the plain-quote and namespacing notes above).
    """
    arguments: dict = {}
    cursor = _Cursor(args_body)

    while True:
        cursor.skip_whitespace()

        if cursor.at_end():
            break

        key = _parse_key(cursor)
        cursor.skip_whitespace()

        if not key or cursor.at_end() or cursor.peek() != ":":
            break

        cursor.position += 1
        arguments[key] = _parse_value(cursor)
        cursor.skip_whitespace()

        if not cursor.at_end() and cursor.peek() == ",":
            cursor.position += 1

    return arguments


def _scrub_pipe_tokens_deep(value):
    """strip_pipe_tokens over every string in a parsed argument tree."""
    if isinstance(value, str):
        return strip_pipe_tokens(value)

    if isinstance(value, dict):
        return {key: _scrub_pipe_tokens_deep(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_scrub_pipe_tokens_deep(item) for item in value]

    return value


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
    # Recurses: since the parser now returns containers, a string nested inside one
    # needs the same scrub a top-level string gets.
    values = _scrub_pipe_tokens_deep(_parse_arguments(body[brace_open + 1:brace_close]))
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


def _render_value(value, escape_keys: bool = False) -> str:
    """
    Render one value in Gemma's trained grammar (format_argument in the canonical
    template). Strings take the <|"|> delimiter, None/bools are bare literals, and
    containers RECURSE rather than collapsing to raw JSON -- an array must render
    [<|"|>a<|"|>,<|"|>b<|"|>], not ["a", "b"], or every call carrying a non-scalar
    parameter puts untrained tokens in front of the model.

    escape_keys wraps object KEYS in the delimiter. Argument bodies pass False (bare
    keys); tool declarations pass True on their free-form branch, mirroring the
    template's default.

    bool is checked before the numeric fallback because bool is an int subclass in
    Python -- json.dumps(True) would otherwise be reached and emit "true" by luck
    for the wrong reason.
    """
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return _render_string_value(value)

    if isinstance(value, dict):
        parts = []

        for key, item in sorted(value.items(), key=lambda entry: entry[0]):
            rendered_key = _render_string_value(key) if escape_keys else key
            parts.append(f"{rendered_key}:{_render_value(item, escape_keys)}")

        return "{" + ",".join(parts) + "}"

    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render_value(item, escape_keys) for item in value) + "]"

    return json.dumps(value)


def _render_gemma_args(values: dict) -> str:
    """
    Render a trained `key:value,...` argument body with keys in SORTED order.

    NO whitespace around ':' or ',' -- the trained format has none, and in a tokenized
    grammar `key: value` and `key:value` are different tokens. Sorting is the parity
    contract: the canonical template renders every argument body through Jinja's
    `| dictsort`, and the C++ twin (Gemma.Protocol.ixx renderArguments) sorts implicitly
    because nlohmann::json is a std::map. Python dicts preserve insertion order, so
    without the sort the two implementations emitted different prompts for identical
    input -- format_tool_response built {result, error} and sent `result,error` where
    C++ sent `error,result`. Pinned by test_gemma_protocol.py and its C++ counterpart.
    """
    parts = []

    for key, value in sorted(values.items(), key=lambda item: item[0]):
        parts.append(f"{key}:{_render_value(value)}")

    return ",".join(parts)


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
        # `value` (not `result`) is what the canonical template emits for a
        # non-mapping response.
        body = f"value:{_render_string_value(result)}"

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
