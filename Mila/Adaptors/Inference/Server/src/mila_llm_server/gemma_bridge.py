"""
The thin layer between MIS's wire shapes and Gemma 4's grammar. It holds no grammar itself.

Every token and every rule lives in the runtime (Dnn.Models.GemmaProtocol) and arrives here
through the mila binding. What is left is the part that is genuinely a server's: which tools a
harness should advertise at all, the correlation id the Anthropic wire requires, and deciding
what a malformed call should do to a live request.

Named for what it is, deliberately. A file called gemma_protocol.py would be the copy of a
grammar the library already owns -- which is what it was, for 856 lines, until the template, the
declaration renderer and answer extraction came down into the runtime beside the value grammar
already there.
"""
import hashlib
import json
import logging

import mila

_log = logging.getLogger(__name__)

#: Gemma's control tokens, reported by the runtime. Fetched once -- they are constants of the
#: checkpoint vocabulary, not configuration -- and never written down here.
TOKENS = mila.gemma_protocol_tokens()

#: Harness and UI tools the model should never be asked to emit a call for. A server's policy
#: about its own client, not a property of the model, so it is applied here rather than in the
#: runtime's renderer -- which advertises whatever it is handed.
_EXCLUDED_TOOLS = {
    "update_plan", "request_user_input", "view_image",
    "spawn_agent", "send_input", "resume_agent", "wait_agent", "close_agent",
}


def tool_declarations(tools: list[dict]) -> str:
    """
    Advertise tools in Gemma's trained declaration grammar, minus the ones a harness runs itself.

    The rendering is the runtime's; the exclusion is ours. Returns '' when nothing is left to
    advertise, and that absence is what tells the model there are no tools.
    """
    if not tools:
        return ""

    advertised = []

    for tool in tools:
        if tool.get("type") not in (None, "function"):
            continue

        declaration = tool.get("function") or tool
        name = declaration.get("name")

        if not name or name in _EXCLUDED_TOOLS:
            continue

        advertised.append(tool)

    if not advertised:
        return ""

    return mila.gemma_tool_declarations(json.dumps(advertised))


def format_prompt(system_block: str, turns: list[dict], continue_open: bool = False) -> str:
    """
    Render a prompt from a system block and the adapters' role-tagged turn list.

    The adapters speak the PROMPT's vocabulary -- a turn is tagged "model", because that is what
    Gemma calls the assistant -- while the binding speaks the conversation's, where the same turn
    is "assistant". Reconciling that here is the whole point of a bridge: there are four adapter
    call sites and one of these, and the last time a wire-shape difference was reconciled in the
    adapters instead, one of the four was missed and returned a live 500.

    system_block already carries whatever tool_declarations returned, because the adapters
    assemble their instructions and their declarations together.
    """
    history = []

    if system_block:
        history.append({"role": "system", "content": system_block})

    for turn in turns:
        history.append({
            "role": "assistant" if turn["role"] == "model" else turn["role"],
            "content": turn["content"],
        })

    return mila.gemma_format_prompt(history, "", continue_open)


def format_tool_call(name: str, arguments_json: str) -> str:
    """One assistant tool call rendered back into the native grammar, for replay."""
    return mila.gemma_format_tool_call(name, arguments_json)


def format_tool_response(name: str, result: str) -> str:
    """A client-executed tool result in the native tool-response grammar."""
    return mila.gemma_format_tool_response(name, result)


def _make_call_id(name: str, arguments: str) -> str:
    """
    A correlation id derived from the call itself, so it is STABLE across re-parses.

    Deliberately a digest rather than a counter: the Responses path is stateless and re-reads a
    transcript on every request, so the same call has to yield the same call_id each time or a
    client's tool result stops matching the call it answers. Qwen's bridge can use a counter
    because nothing there re-parses.
    """
    digest = hashlib.sha256(f"{name}\x00{arguments}".encode()).hexdigest()[:24]

    return f"call_{digest}"


def parse_tool_call(text: str) -> dict | None:
    """
    The most recent tool call in a raw Gemma turn as {'call_id', 'name', 'arguments'}, or None.

    The runtime returns only {'name', 'arguments'} -- Gemma's grammar has no slot for an id -- so
    the call_id every adapter here reads is minted at this seam, which is where wire bookkeeping
    belongs rather than in something the model said.
    """
    call = mila.gemma_parse_tool_call(text)

    if call is None:
        return None

    return {
        "call_id": _make_call_id(call["name"], call["arguments"]),
        "name": call["name"],
        "arguments": call["arguments"],
    }


def answer_text(text: str) -> str:
    """Reduce a raw turn to what a user should see."""
    return mila.gemma_extract_answer(text)


def strip_control_tokens(text: str) -> str:
    """Every registered control token removed from decoded text."""
    return mila.gemma_strip_control_tokens(text)
