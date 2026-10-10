# protocols/utils.py
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# The identifier clients see is config.loaded.name, read per request at the sites that
# emit it. It was a constant here, bound to settings.model at import -- before the worker
# had resolved anything -- so a store match that differed in case (the store matches
# case-insensitively) reported a name that was not the one loaded.

def parse_stop(stop: str | list[str] | None) -> list[str]:
    """OpenAI's `stop` is a string, a list of strings, or absent; empty strings match nowhere."""
    if stop is None:
        return []

    if isinstance(stop, str):
        stop = [stop]

    return [sequence for sequence in stop if sequence]


def truncate_at_stop(text: str, stop: list[str]) -> tuple[str, str | None]:
    """
    Cut text at the earliest occurrence of any stop sequence. Returns the text before it and
    the sequence found, or None. Earliest, not first in the list: OpenAI stops generation at the
    first sequence produced, wherever it sits in the list.
    """
    found = [(text.find(sequence), sequence) for sequence in stop]
    found = [(position, sequence) for position, sequence in found if position >= 0]

    if not found:
        return text, None

    position, sequence = min(found)

    return text[:position], sequence


class StopScanner:
    """
    Applies stop sequences to a streamed reply, as truncate_at_stop applies them to a whole one.

    A sequence can arrive split across chunks, so the tail that could still begin one is held back
    until the next chunk settles it. feed() returns the text that is safe to send; once a sequence
    is found, `matched` names it and nothing after it is ever returned. flush() releases what is
    held when the reply ends without one.
    """

    def __init__(self, stop: list[str]):
        self._stop = stop
        self._held = ""
        self._hold = max((len(sequence) for sequence in stop), default=1) - 1
        self.matched: str | None = None

    def feed(self, text: str) -> str:
        if self.matched is not None:
            return ""

        pending = self._held + text
        before, matched = truncate_at_stop(pending, self._stop)

        if matched is not None:
            self.matched = matched
            self._held = ""

            return before

        cut = max(len(pending) - self._hold, 0)

        while cut < len(pending) and not any(sequence.startswith(pending[cut:]) for sequence in self._stop):
            cut += 1

        self._held = pending[cut:]

        return pending[:cut]

    def flush(self) -> str:
        held, self._held = self._held, ""

        return held


#: Why a reply ended, as the routes carry it. Each protocol adapter spells these its own way.
END_TURN = "end_turn"
STOP_SEQUENCE = "stop_sequence"
MAX_TOKENS = "max_tokens"
CONTEXT_LIMIT = "context_limit"


def finish_reason_from_status(status: str, requested_tokens: int, allowed_tokens: int) -> str:
    """
    The reason a reply ended, from the binding's generate() status. The routes lower a request's
    max_tokens to what the context has left, so a reply stopped at that lowered budget ran out of
    context, not of the tokens it asked for. A cancelled generation is the model's turn ending: the
    worker cancels at a protocol marker, and a route that cancels for a stop sequence or a dropped
    client sets the reason itself.
    """
    if status == "length":
        return CONTEXT_LIMIT if allowed_tokens < requested_tokens else MAX_TOKENS

    if status == "context_limit":
        return CONTEXT_LIMIT

    return END_TURN


def openai_finish_reason(reason: str) -> str:
    """OpenAI's spelling. It has no value for a full context, and `length` is the one a client handles."""
    return "length" if reason in (MAX_TOKENS, CONTEXT_LIMIT) else "stop"


def anthropic_stop_reason(reason: str) -> str:
    return {
        END_TURN: "end_turn",
        STOP_SEQUENCE: "stop_sequence",
        MAX_TOKENS: "max_tokens",
        CONTEXT_LIMIT: "model_context_window_exceeded",
    }[reason]


def parse_completion_prompt(prompt: str | list) -> tuple[str, list[int]]:
    """
    Read a Completions `prompt` as either text or token ids, returning (text, ids) with exactly
    one of them set. OpenAI accepts a string, a list of token ids, or a list of either; a list
    of prompts is a batch, and only a batch of one is served.

    Token ids are passed to the model untouched. That is what an evaluation harness relies on:
    it renders and tokenizes a prompt once, sends the same ids to every engine it compares, and
    the prompt template and tokenizer drop out of the comparison.
    """
    if isinstance(prompt, str):
        return prompt, []

    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a string, a list of token ids, or a list holding one of either")

    if all(isinstance(token, int) for token in prompt):
        return "", list(prompt)

    if len(prompt) != 1:
        raise ValueError(f"prompt holds a batch of {len(prompt)}; this server completes one prompt per request")

    return parse_completion_prompt(prompt[0])


def extract_content(content: str | list) -> str:
    if isinstance(content, str):
        return content
    # Responses API item content blocks: user turns carry "input_text", prior
    # assistant turns carry "output_text". Both must be extracted or the model's
    # own history collapses to empty turns.
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("input_text", "text", "output_text")
    ]
    return "".join(text_parts)
