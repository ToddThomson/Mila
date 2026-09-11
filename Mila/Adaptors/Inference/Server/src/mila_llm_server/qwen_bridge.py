"""
The thin layer between MIS's wire shapes and Qwen's grammar. It holds no grammar itself.

Every token and every rule lives in the runtime (Dnn.Components.QwenProtocol) and arrives here
through the mila binding. What is left is the part that is genuinely a server's: deciding what a
malformed call should do to a live request, and reducing a raw turn to display text.

Named for what it is, deliberately. A file called qwen_protocol.py would be the third copy of a
grammar the library already owns, which is exactly what this design exists to prevent.
"""
import itertools
import logging

import mila

_log = logging.getLogger(__name__)

#: Correlation ids for tool calls. The runtime does not mint these and should not: Qwen's
#: template pairs a call to its result positionally and never renders an id, so it is the
#: server's own bookkeeping, needed here because the Anthropic wire requires tool_use.id.
_call_ids = itertools.count(1)

#: Qwen's control tokens, reported by the runtime. Fetched once -- they are constants of the
#: checkpoint vocabulary, not configuration -- and never written down here.
TOKENS = mila.qwen_protocol_tokens()


def parse_tool_call(text: str) -> dict | None:
    """
    The first tool call in a raw Qwen turn as {'call_id', 'name', 'arguments'}, or None.

    The runtime returns only {'name', 'arguments'} -- Qwen's template pairs a call to its result
    positionally and renders no id -- so the call_id every adapter here reads is minted at this
    seam. That is the right place for it: it is wire bookkeeping the Anthropic protocol needs,
    not something the model said.

    The runtime raises when a <tool_call> span holds something that is not a call, because that
    is the model failing at its own protocol and is worth seeing. A server cannot let that end
    the request: the turn still has an answer in it, and a 500 would lose it. So it is logged and
    the turn degrades to prose -- visible to an operator, survivable for the caller.
    """
    try:
        call = mila.qwen_parse_tool_call(text)
    except RuntimeError as error:
        _log.warning("Qwen emitted a malformed tool call, treating the turn as prose: %s", error)
        return None

    if call is None:
        return None

    return {
        "call_id": f"call_{next(_call_ids)}",
        "name": call["name"],
        "arguments": call["arguments"],
    }


def answer_text(text: str) -> str:
    """
    Reduce a raw turn to what a user should see.

    Two things come off. A tool-call span is protocol, not prose, and the model is told it may
    reason in natural language BEFORE a call but not after -- so everything from the span onward
    goes. And a turn generated with thinking on opens inside the reasoning span, so it carries a
    CLOSING marker with no opening one; that close and everything before it is the reasoning.
    """
    reasoning_close = TOKENS["reasoning_close"]
    closed_at = text.find(reasoning_close)

    if closed_at != -1:
        text = text[closed_at + len(reasoning_close):]

    call_at = text.find(TOKENS["tool_call_open"])

    if call_at != -1:
        text = text[:call_at]

    return text.strip()
