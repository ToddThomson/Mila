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


def truncate_at_stop(text: str, stop: list[str]) -> tuple[str, bool]:
    """
    Cut text at the earliest occurrence of any stop sequence. Returns the text before it and
    whether one was found. Earliest, not first in the list: OpenAI stops generation at the
    first sequence produced, wherever it sits in the list.
    """
    positions = [text.find(sequence) for sequence in stop]
    found = [position for position in positions if position >= 0]

    if not found:
        return text, False

    return text[:min(found)], True


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
