# protocols/utils.py
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# The identifier clients see is config.loaded.name, read per request at the sites that
# emit it. It was a constant here, bound to settings.model at import -- before the worker
# had resolved anything -- so a store match that differed in case (the store matches
# case-insensitively) reported a name that was not the one loaded.

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
