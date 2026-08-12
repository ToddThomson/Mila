"""
Assembles instruct-format prompts from chat messages.

Dispatches on the configured model family:
  - llama: Llama 3.x instruct template (<|start_header_id|> ... <|eot_id|>)
  - gemma: Gemma 4 instruct template (<|turn> ... <turn|>)

The Gemma checkpoint Mila ships uses the <|turn>/<turn|> control tokens (NOT the
Gemma 3-style <start_of_turn>/<end_of_turn>, which are absent from its vocabulary),
roles system/user/model, a <bos> prefix, and a <|turn>model\\n generation primer.
Reference: Mila chat harness formatGemmaPrompt() and
https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
"""

import json

from mila_llm_server.config import loaded, ModelFamily

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# --- Llama 3.x instruct tokens ---
_LLAMA_BOS = "<|begin_of_text|>"
_LLAMA_HEADER_START = "<|start_header_id|>"
_LLAMA_HEADER_END = "<|end_header_id|>"
_LLAMA_EOT = "<|eot_id|>"

# --- Gemma 4 instruct tokens (Mila checkpoint) ---
_GEMMA_BOS = "<bos>"
_GEMMA_TURN_OPEN = "<|turn>"
_GEMMA_TURN_CLOSE = "<turn|>"
# Thinking is off for coding/agentic harnesses. Priming an empty thought channel
# suppresses "ghost" thought sections the 12B otherwise emits when deactivated.
_GEMMA_THOUGHT_PRIME = "<|channel>thought\n<channel|>"


def build_instruct_prompt(
    user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history: list[dict[str, str]] | None = None,
    tools: list[dict] | None = None,
) -> str:
    """
    Build an instruct prompt string ready for tokenization, in the template of the
    configured model family. Each history entry must have 'role' (user | assistant |
    system) and 'content'. The returned string ends with the assistant/model primer
    so the model generates the response turn directly.
    """
    if loaded.family == ModelFamily.gemma:
        return _build_gemma_prompt(user_message, system_prompt, history, tools)

    return _build_llama_prompt(user_message, system_prompt, history, tools)


def _build_llama_prompt(
    user_message: str,
    system_prompt: str,
    history: list[dict[str, str]] | None,
    tools: list[dict] | None,
) -> str:
    parts: list[str] = [_LLAMA_BOS]

    if system_prompt or tools:
        system_block = system_prompt or ""

        if tools:
            tools_str = json.dumps(tools, indent=2)
            system_block = f"{system_block}\n\nYou have access to the following tools:\n{tools_str}"

        parts.append(f"{_LLAMA_HEADER_START}system{_LLAMA_HEADER_END}\n\n{system_block}{_LLAMA_EOT}")

    for turn in (history or []):
        role = turn["role"]
        content = turn["content"]
        parts.append(f"{_LLAMA_HEADER_START}{role}{_LLAMA_HEADER_END}\n\n{content}{_LLAMA_EOT}")

    parts.append(f"{_LLAMA_HEADER_START}user{_LLAMA_HEADER_END}\n\n{user_message}{_LLAMA_EOT}")
    parts.append(f"{_LLAMA_HEADER_START}assistant{_LLAMA_HEADER_END}\n\n")

    return "".join(parts)


def _gemma_turn(role: str, content: str) -> str:
    """Render one Gemma turn: <|turn>{role}\\n{content}<turn|>\\n."""
    return f"{_GEMMA_TURN_OPEN}{role}\n{content}{_GEMMA_TURN_CLOSE}\n"


def _build_gemma_prompt(
    user_message: str,
    system_prompt: str,
    history: list[dict[str, str]] | None,
    tools: list[dict] | None,
) -> str:
    parts: list[str] = [_GEMMA_BOS]

    # Gemma has no native system role; it collapses to a single leading system turn.
    if system_prompt or tools:
        system_block = system_prompt or ""

        if tools:
            tools_str = json.dumps(tools, indent=2)
            system_block = f"{system_block}\n\nYou have access to the following tools:\n{tools_str}"

        parts.append(_gemma_turn("system", system_block))

    for turn in (history or []):
        role = turn["role"]

        if role == "system":
            continue

        # The assistant is the "model" role in Gemma's template.
        gemma_role = "model" if role == "assistant" else "user"
        parts.append(_gemma_turn(gemma_role, turn["content"]))

    parts.append(_gemma_turn("user", user_message))

    # Generation primer + thinking-off channel prime.
    parts.append(f"{_GEMMA_TURN_OPEN}model\n")
    parts.append(_GEMMA_THOUGHT_PRIME)

    return "".join(parts)
