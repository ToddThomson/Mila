"""
OpenAI Models API protocol adapter.
  Models:  GET /v1/models
"""
import time

from protocols.base import ModelsCapable
from protocols.utils import MODEL_NAME
from config import settings


class OpenAIModelsAdapter(ModelsCapable):

    @property
    def models_path(self) -> str:
        return "/v1/models"

    def format_models_response(self) -> dict:
        created = int(time.time())

        card = {
            "id": MODEL_NAME,
            "object": "model",
            "created": created,
            "owned_by": "mila",
            "context_window": settings.context_length,
            "max_output_tokens": settings.default_max_new_tokens,
            "supports_parallel_tool_calls": False,
            "supports_reasoning": False,
            "reasoning_summary_format": "none",
            "slug": MODEL_NAME,
            "display_name": MODEL_NAME,
            "capabilities": {
                "tools": True,
                "apply_patch": True,
                "exec_command": True,
            },
        }

        return {
            "object": "list",
            "data": [card],
        }