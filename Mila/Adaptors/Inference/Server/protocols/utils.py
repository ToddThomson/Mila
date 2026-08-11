# protocols/utils.py
from config import settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# The store name MIS was configured with, which is also the identifier reported to
# clients: one name, so what a client sees in /v1/models is what it can install.
MODEL_NAME = settings.model

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
