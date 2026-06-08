# protocols/utils.py
from config import settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MODEL_NAME = settings.model_name

def extract_content(content: str | list) -> str:
    if isinstance(content, str):
        return content
    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("input_text", "text")
    ]
    return "".join(text_parts)
