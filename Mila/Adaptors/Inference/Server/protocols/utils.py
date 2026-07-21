# protocols/utils.py
from config import settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MODEL_NAME = settings.model_name

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
