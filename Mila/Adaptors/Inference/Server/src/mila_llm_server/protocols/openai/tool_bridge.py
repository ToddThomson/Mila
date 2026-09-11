"""
Bridges the gap between Codex CLI's OpenAI tool schemas and the simplified
[func_name(params)] format that Llama 3.2 3B instruct was fine-tuned on.

Inbound:  Codex sends full OpenAI JSON tool definitions.
          build_tool_injection() distills them into a system prompt suffix
          the 3B model can reliably act on.

Outbound: The model emits <|python_tag|>[func_name(key="value", ...)]
          parse_tool_call() detects and parses that into an OpenAI
          Responses API function_call output item Codex expects.

Detection order mirrors Chat.ToolCallParser:
  1. <|python_tag|> prefix -> extract bracketed pythonic call
  2. Leading '[' without prefix -> bare pythonic call fallback (requires
     the bracket content to begin with a valid identifier character to
     suppress false positives such as list literals in prose).

apply_patch notes
-----------------
apply_patch is NOT sent as a function tool by the current Codex CLI.
File changes are made exclusively through exec_command by placing an
apply_patch invocation in the cmd string. Codex's UnifiedExecHandler
intercepts cmd before spawning any shell and applies the patch natively.

The two cmd shapes Codex intercepts:

  Direct argv (preferred, what gpt-5.x Codex models emit):
    apply_patch "*** Begin Patch\\n*** Add File: path/to/file\\n+content\\n*** End Patch"

  Heredoc (for larger patches):
    apply_patch << 'EOF'\\n*** Begin Patch\\n...\\n*** End Patch\\nEOF

The bridge teaches the 3B model the direct argv form via an example in
_LLAMA_TOOL_HEADER. Because the model emits \\n inside the quoted cmd
string, _parse_pythonic_params unescapes standard escape sequences in
quoted values so that Codex receives real newlines in the cmd string.

Patch format grammar (from codex-rs/apply-patch/src/parser.rs):
  *** Begin Patch
  *** Add File: <path>          creates a new file
  +<line>                       line to add
  *** Delete File: <path>       removes a file
  *** Update File: <path>       modifies an existing file
  *** Move to: <new_path>       optional rename inside an Update hunk
  @@ <context anchor>           optional region anchor
   <context line>               space-prefixed unchanged line
  -<old line>                   line to remove
  +<new line>                   replacement line
  *** End of File               optional, targets end-of-file region
  *** End Patch
"""

import hashlib
import json
import re
import uuid

# Tools excluded from injection entirely. These are not expected to be
# called by the 3B model and their argument structures cannot be expressed
# in flat key=value pairs.
# apply_patch is intentionally NOT here: it is not sent as a function tool
# by the current Codex CLI, so it never appears in tools[] and requires no
# injection entry. File changes go through exec_command instead.
_EXCLUDED_TOOLS = {
    "update_plan", "request_user_input",
    "spawn_agent", "send_input", "resume_agent",
    "wait_agent", "close_agent", "view_image",
}

# Per-tool allowlist of parameter names shown in the injection.
# Parameters absent from the allowlist are silently suppressed. This
# prevents array-typed sibling parameters (e.g. exec_command.prefix_rule)
# from leaking the argv-array mental model into cmd output.
# A tool with no entry here shows all scalar params (default behaviour).
_TOOL_PARAM_ALLOWLIST: dict[str, set[str]] = {
    "exec_command": {"cmd", "workdir"},
}

# Scalar parameter types the injection builder can describe meaningfully.
# Non-scalar types (object, array) are annotated as (JSON string) instead.
_SCALAR_TYPES = {"string", "number", "integer", "boolean"}

# Correct known tokenization artifacts where the model prepends a stray
# character or misspells the tool name.
_TOOL_NAME_CORRECTIONS: dict[str, str] = {
    "eexec_command": "exec_command",
    "exec_commmand": "exec_command",
    "exec_comand":   "exec_command",
}

_PYTHON_TAG = "<|python_tag|>"

# Regex that matches a valid Python identifier followed by '(' at the
# start of a string. Used by the bare-bracket fallback to suppress false
# positives such as list literals in prose (e.g. "options: [A, B, C]").
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(")

# Standard escape sequences unescaped in quoted param values.
# Order matters: \\\\ must be last so prior replacements are not re-processed.
_UNESCAPE_TABLE = [
    ("\\n",  "\n"),
    ("\\t",  "\t"),
    ("\\r",  "\r"),
    ('\\"',  '"'),
    ("\\\\", "\\"),
]

_LLAMA_TOOL_HEADER = (
    "\n\nYou have access to tools to complete tasks. "
    "When you need to use a tool, you MUST output ONLY the tool call and nothing else. "
    "Use EXACTLY this format:\n"
    "<|python_tag|>[func_name(param1=\"value1\", param2=\"value2\")]\n\n"
    "Only use the tools listed below. Do not reference or call any other tools.\n\n"
    "Examples:\n"
    '<|python_tag|>[exec_command(cmd="ls -la")]\n'
    '<|python_tag|>[exec_command(cmd="g++ -o sum sum.cpp")]\n'
    '<|python_tag|>[exec_command(cmd="apply_patch \\"*** Begin Patch\\n*** Add File: hello.cpp\\n+#include <iostream>\\n+int main() { return 0; }\\n*** End Patch\\"")]\n\n'
    "Do NOT use JSON arrays for cmd. Always use a plain shell string:\n"
    '  WRONG: [exec_command(cmd=["g++", "-o", "sum", "sum.cpp"])]\n'
    '  RIGHT: [exec_command(cmd="g++ -o sum sum.cpp")]\n\n'
    "Do NOT write file content with echo or printf. Use apply_patch via exec_command:\n"
    '  WRONG: [exec_command(cmd="echo \'#include <iostream>\' > main.cpp")]\n'
    '  RIGHT: [exec_command(cmd="apply_patch \\"*** Begin Patch\\n*** Add File: main.cpp\\n+#include <iostream>\\n*** End Patch\\"")]\n\n'
    "Available tools:\n"
)

# The Codex CLI system prompt instructs the model to call apply_patch as a
# dedicated tool ("Use the `apply_patch` tool to edit files"). For gpt-5.x
# models that instruction is correct: apply_patch is a first-class Responses
# API built-in that never appears in tools[]. For the 3B Llama model running
# in the MIS there is no such built-in; file changes must go through
# exec_command with an apply_patch shell invocation intercepted natively by
# Codex's UnifiedExecHandler. This block is appended after the tool list so
# it appears last in the assembled system prompt and overrides the Codex
# instruction with the highest positional weight.
_APPLY_PATCH_RECONCILIATION = (
    "\n\nIMPORTANT OVERRIDE: The apply_patch built-in tool is NOT available "
    "in this environment. Ignore any prior instruction to call apply_patch "
    "directly as a tool. To create or modify files you MUST use exec_command "
    "with an apply_patch shell invocation as shown in the examples above:\n"
    "<|python_tag|>[exec_command(cmd=\"apply_patch \\\"*** Begin Patch\\n"
    "*** Add File: path/to/file\\n"
    "+file content here\\n"
    "*** End Patch\\\"\")]\n"
    "This is the ONLY correct method for all file creates, updates, and deletes."
)


def build_tool_injection(tools: list[dict]) -> str:
    """
    Distill a list of OpenAI function tool definitions into a Llama-compatible
    system prompt suffix.

    Tools in _EXCLUDED_TOOLS are skipped entirely. For included tools,
    _TOOL_PARAM_ALLOWLIST controls which parameters are shown; parameters
    absent from the allowlist are silently suppressed. Non-scalar parameters
    that do survive (object, array) are annotated as (JSON string) so the
    model knows to serialise them.
    """
    if not tools:
        return ""

    lines: list[str] = []

    for tool in tools:
        if tool.get("type") != "function":
            continue

        fn = tool.get("function") or tool
        name = fn.get("name", "")

        if name in _EXCLUDED_TOOLS:
            continue

        desc = fn.get("description", "")
        params_schema = fn.get("parameters", {})
        props = params_schema.get("properties", {})

        allowlist = _TOOL_PARAM_ALLOWLIST.get(name)
        if allowlist is not None:
            props = {k: v for k, v in props.items() if k in allowlist}

        param_summaries = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "string")
            pdesc = pdef.get("description", "")

            if ptype in _SCALAR_TYPES:
                display_type = ptype
            else:
                display_type = "JSON string"

            param_summaries.append(f"  {pname} ({display_type}): {pdesc}")

        lines.append(f"- {name}: {desc}")
        if param_summaries:
            lines.extend(param_summaries)

    if not lines:
        return ""

    return _LLAMA_TOOL_HEADER + "\n".join(lines) + _APPLY_PATCH_RECONCILIATION


def _find_closing_quote(s: str, start: int) -> int:
    """
    Scan forward from start in s looking for an unescaped double-quote.
    Backslash sequences are consumed as a unit so that \\\" does not
    terminate the value prematurely. Returns the index of the closing
    quote, or -1 if none is found.
    """
    i = start
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == '"':
            return i
        i += 1
    return -1


def _find_closing_paren(s: str, start: int) -> int:
    """
    Scan forward from start in s looking for the closing ')' that matches
    the implicit open at the call site. Quoted strings are consumed as a
    unit (respecting backslash escapes) so that a ')' inside a shell
    command such as echo $(date) does not terminate the parameter list
    prematurely. Returns the index of the closing paren, or -1 if none
    is found.
    """
    i = start
    depth = 1

    while i < len(s):
        ch = s[i]

        if ch == "\\":
            i += 2
            continue

        if ch == '"':
            close = _find_closing_quote(s, i + 1)
            if close == -1:
                return -1
            i = close + 1
            continue

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def _find_closing_bracket(s: str, open_ch: str, close_ch: str, start: int) -> int:
    """
    Scan forward from start in s looking for the close_ch that balances
    the implicit open_ch at the call site. Quoted strings (respecting
    backslash escapes) and nested bracket pairs are consumed as a unit.
    Returns the index of the closing bracket, or -1 if none is found.

    Used to capture array/object-valued parameters (e.g. cmd=["g++", ...])
    as a raw string rather than stopping at the first interior comma.
    """
    i = start
    depth = 1

    while i < len(s):
        ch = s[i]

        if ch == "\\":
            i += 2
            continue

        if ch == '"':
            close = _find_closing_quote(s, i + 1)
            if close == -1:
                return -1
            i = close + 1
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return -1


def _unescape_value(value: str) -> str:
    """
    Unescape standard backslash sequences in a quoted param value.

    The model emits \\n, \\t, \\r, \\\", and \\\\ inside quoted strings.
    After extraction the characters are still raw (e.g. backslash + 'n').
    This step converts them to real characters so that the newlines in an
    apply_patch cmd string arrive as actual newlines at Codex, which is
    required for UnifiedExecHandler to intercept the patch correctly.
    """
    result = value
    for escaped, real in _UNESCAPE_TABLE:
        result = result.replace(escaped, real)
    return result


def _extract_bracket_call(text: str) -> tuple[str, str] | None:
    """
    Extract (func_name, raw_params) from the first [func_name(...)] block
    in text.

    The closing ']' is located by scanning forward from the paren-balanced
    end of the argument list rather than via rfind, so that shell commands
    containing ']' or ')' do not corrupt the parse. The bare-bracket
    fallback path additionally requires the bracket content to begin with a
    valid identifier followed by '(' to suppress false positives such as
    list literals in prose (e.g. "options: [A, B, C]").

    Mirrors Chat.ToolCallParser::parsePythonicCall.
    """
    start = text.find("[")
    if start == -1:
        return None

    inner_start = start + 1

    # Bare-bracket fallback guard: content must look like an identifier call.
    if not _IDENTIFIER_RE.match(text[inner_start:]):
        return None

    paren_open = text.find("(", inner_start)
    if paren_open == -1:
        return None

    func_name = text[inner_start:paren_open]

    paren_close = _find_closing_paren(text, paren_open + 1)
    if paren_close == -1:
        return None

    raw_params = text[paren_open + 1:paren_close]

    # Expect a closing ']' after the closing ')'.
    bracket_close = text.find("]", paren_close + 1)
    if bracket_close == -1:
        return None

    return func_name, raw_params


def _parse_pythonic_params(raw_params: str) -> dict:
    """
    Parse key="value" or key=value param pairs from a pythonic call argument
    string. Mirrors the logic in Chat.ToolCallParser::parsePythonicCall.

    Quoted values:
      Terminated by the first unescaped double-quote after the opening
      delimiter (via _find_closing_quote). Standard escape sequences are
      then unescaped via _unescape_value so that \\n becomes a real newline,
      which is required for apply_patch cmd strings to be correctly
      intercepted by Codex's UnifiedExecHandler.

    Array/object values (e.g. cmd=["g++", "-o", "sum"]):
      Captured in their entirety by _find_closing_bracket rather than
      stopping at the first interior comma. The raw literal (including
      brackets) is stored as the value string so the argument dict is
      correctly delimited, and Codex can report a useful error rather than
      receiving a silently truncated value.

    Unquoted scalar values:
      Terminated by the next comma or end of string.
    """
    args: dict = {}
    param = raw_params.strip()

    while param:
        eq = param.find("=")
        if eq == -1:
            break

        key = param[:eq].strip()
        param = param[eq + 1:]

        if param.startswith('"'):
            close = _find_closing_quote(param, 1)
            if close == -1:
                break
            raw_value = param[1:close]
            value = _unescape_value(raw_value)
            param = param[close + 1:].lstrip(", \t")

        elif param.startswith("["):
            close = _find_closing_bracket(param, "[", "]", 1)
            if close == -1:
                break
            value = param[:close + 1]
            param = param[close + 1:].lstrip(", \t")

        elif param.startswith("{"):
            close = _find_closing_bracket(param, "{", "}", 1)
            if close == -1:
                break
            value = param[:close + 1]
            param = param[close + 1:].lstrip(", \t")

        else:
            comma = param.find(",")
            if comma == -1:
                value = param.strip()
                param = ""
            else:
                value = param[:comma].strip()
                param = param[comma + 1:].lstrip(" \t")

        args[key] = value

    return args


def _make_call_id(func_name: str, arguments: str) -> str:
    """
    Derive a deterministic call_id from the function name and serialised
    arguments so that retrying the same model output yields the same
    call_id. Codex uses call_id to correlate function_call_output items
    back to their originating function_call; a stable id makes retries
    idempotent from Codex's perspective.
    """
    digest_input = f"{func_name}\x00{arguments}"
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:24]
    return f"call_{digest}"


def parse_tool_call(
    text: str,
    known_tools: set[str] | None = None,
) -> dict | None:
    """
    Detect and parse a tool call from model output.

    Checks for <|python_tag|> prefix first (the format Llama 3.2 3B
    produces), then falls back to a bare [func_name(...)] match. The
    bare-bracket path requires the bracket content to begin with a valid
    identifier to suppress false positives.

    Applies name corrections for known tokenization artifacts. If
    known_tools is supplied, returns None for any func_name not present
    in that set so that hallucinated tool names do not propagate to Codex
    as plausible-looking function_call items.

    call_id is derived deterministically from (func_name, arguments) so
    that retrying the same model output is idempotent from Codex's
    perspective. id remains a random uuid4 as it is a session-scoped item
    identifier, not a correlation key.

    The returned shape matches the Responses API function_call item exactly:
      type, id, call_id, name, arguments — no extra fields.
    """
    search_text = text
    if _PYTHON_TAG in text:
        search_text = text[text.index(_PYTHON_TAG) + len(_PYTHON_TAG):]

    result = _extract_bracket_call(search_text)
    if result is None:
        return None

    func_name, raw_params = result
    func_name = func_name.strip()
    func_name = _TOOL_NAME_CORRECTIONS.get(func_name, func_name)

    if known_tools is not None and func_name not in known_tools:
        return None

    args = _parse_pythonic_params(raw_params)
    arguments = json.dumps(args)
    call_id = _make_call_id(func_name, arguments)

    return {
        "type": "function_call",
        "id": f"fc_{uuid.uuid4().hex}",
        "call_id": call_id,
        "name": func_name,
        "arguments": arguments,
    }