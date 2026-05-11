"""
Assembles Llama 3.2 instruct-format prompts from chat messages.

Template reference:
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
  <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>\n\n
"""

_BOS = "<|begin_of_text|>"
_HEADER_START = "<|start_header_id|>"
_HEADER_END = "<|end_header_id|>"
_EOT = "<|eot_id|>"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def build_instruct_prompt(
    user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history: list[dict[str, str]] | None = None,
) -> str:
    """
    Build a Llama 3.2 instruct prompt string ready for tokenization.

    Each entry in history must have 'role' (user | assistant) and 'content'.
    The returned string ends with the open assistant header so the model
    generates the response turn directly.
    """
    parts: list[str] = [_BOS]

    if system_prompt or tools:
        system_block = system_prompt or ""
        if tools:
            tools_str = json.dumps(tools, indent=2)
            system_block = f"{system_block}\n\nYou have access to the following tools:\n{tools_str}"
        parts.append(f"{_HEADER_START}system{_HEADER_END}\n\n{system_block}{_EOT}")

    for turn in (history or []):
        role = turn["role"]
        content = turn["content"]
        parts.append(f"{_HEADER_START}{role}{_HEADER_END}\n\n{content}{_EOT}")

    parts.append(f"{_HEADER_START}user{_HEADER_END}\n\n{user_message}{_EOT}")
    parts.append(f"{_HEADER_START}assistant{_HEADER_END}\n\n")

    return "".join(parts)
