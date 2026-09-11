"""
Reference parity: MIS's assembled prompt must be BYTE-IDENTICAL to the output of
Google's canonical chat_template.jinja.

This is the strongest oracle we have, and the only one that is not circular. Every
other test in this suite encodes someone's *reading* of the Gemma 4 format -- so a
misreading produces a confidently green suite over a broken prompt. That is exactly
what happened: nine divergences shipped, several corrupting tool calls, under passing
tests. Here the reference answer comes from Google's own implementation.

The template is vendored UNMODIFIED at tests/reference/. Its source, checksum, and license
are recorded in the repository's root NOTICE.md -- the single place Mila tracks third-party
material. Do not add licensing notes here.

REFRESHING IT: re-download from the URL in NOTICE.md, update the checksum and retrieval
date there, and re-run. **A diff appearing afterwards is a FINDING -- upstream changed the
format and we must follow it -- not a test bug. Never edit the vendored file to make this
test pass; it is not ours, and its value comes entirely from being Google's answer rather
than ours.**

UPSTREAM QUIRK worth knowing before you touch the template: it closes its `parameters`
block from inside the `type:` branch, so a schema with no `type` leaves the block unclosed
with a dangling comma. MIS always closes it -- a deliberate, documented deviation (see
_build_trained_tool_declarations). Do not "fix" the template.

Scope: the Anthropic adapter path (Claude Code). The Codex/Responses adapter shares
assemble_prompt and append_model_tool_span, so it inherits the coverage; a dedicated
case would still be worth adding if that path diverges.
"""
import pathlib

import pytest

jinja2 = pytest.importorskip("jinja2", reason="reference parity needs the jinja2 dev dependency")
from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mila_llm_server.protocols.anthropic.messages import AnthropicMessagesAdapter

TEMPLATE = pathlib.Path(__file__).parent / "reference" / "gemma4_12b_chat_template.jinja"


def render_reference(**context) -> str:
    """
    Render the vendored template the way transformers.apply_chat_template does:
    sandboxed, trim_blocks + lstrip_blocks on, and DEFAULT (non-strict) undefined.

    The undefined mode is load-bearing, not incidental: the template reads optional
    schema keys directly (value['enum']) and relies on a miss being falsy, so
    StrictUndefined raises on a perfectly ordinary tool. Rendering it any other way
    tests a template Google does not ship.
    """
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(TemplateError(message))
    context.setdefault("bos_token", "<bos>")
    context.setdefault("add_generation_prompt", True)
    context.setdefault("enable_thinking", False)
    return env.from_string(TEMPLATE.read_text(encoding="utf-8")).render(**context)


SYSTEM = "You are helpful."

# The same tool, in each side's native shape: OpenAI-ish for the template, Anthropic
# for the adapter. Keep these two in sync -- they must describe the SAME tool or the
# comparison is meaningless.
PARAMETERS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path"},
        "content": {"type": "string", "description": "Body"},
    },
    "required": ["file_path", "content"],
}
REFERENCE_TOOLS = [{"type": "function", "function": {
    "name": "Write", "description": "Write a file", "parameters": PARAMETERS}}]
ANTHROPIC_TOOLS = [{"name": "Write", "description": "Write a file", "input_schema": PARAMETERS}]

CALL_ARGUMENTS = {"file_path": "hello.txt", "content": "hello world"}
TOOL_RESULT = "File created successfully"


def reference_messages(resumed: bool) -> list:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Create hello.txt"}]
    if resumed:
        messages += [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "function": {"name": "Write",
                                                      "arguments": CALL_ARGUMENTS}}]},
            {"role": "tool", "tool_call_id": "c1", "content": TOOL_RESULT},
        ]
    return messages


def anthropic_messages(resumed: bool) -> list:
    messages = [{"role": "user", "content": "Create hello.txt"}]
    if resumed:
        messages += [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "c1", "name": "Write", "input": CALL_ARGUMENTS}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": TOOL_RESULT}]},
        ]
    return messages


def assert_matches_reference(resumed: bool) -> None:
    expected = render_reference(messages=reference_messages(resumed), tools=REFERENCE_TOOLS)
    # Drive the REAL adapter, not a hand-built prompt: routing through the production
    # path is what caught a stray newline in the span joiner.
    actual = AnthropicMessagesAdapter()._build_gemma_prompt(
        anthropic_messages(resumed), SYSTEM, ANTHROPIC_TOOLS)

    assert actual == expected, (
        "MIS prompt diverged from Google's template.\n"
        f"--- REFERENCE ---\n{expected!r}\n--- MIS ---\n{actual!r}"
    )


class TestReferenceParity:
    def test_fresh_turn_matches_reference(self):
        """System turn + trained declarations + user turn + the turn-start thought prime."""
        assert_matches_reference(resumed=False)

    def test_resumed_after_tool_result_matches_reference(self):
        """
        The hard case: the model turn stays OPEN across the tool exchange, the tool
        spans concatenate with no separator, and NOTHING is appended after the tool
        response -- no second thought prime, no trailing newline.
        """
        assert_matches_reference(resumed=True)
