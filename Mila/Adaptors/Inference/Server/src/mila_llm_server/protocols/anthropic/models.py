"""
Anthropic Models API protocol adapter.
  Models:  GET /v1/models

Anthropic's own shape, not OpenAI's: a `data` list of `{type, id, display_name, created_at}`
with the `has_more`/`first_id`/`last_id` pagination envelope, which is what a client written
against the Messages API expects to find beside it.

This endpoint exists here for a second reason. A license that requires displayed attribution
requires it of whoever presents the model, and a server presents its model in exactly two
places: its startup log and its model list. Without this adapter the Anthropic protocol had
only the first, so the duty fell on a log line nobody reads.
"""
from datetime import datetime, timezone

from mila_llm_server.protocols.base import ModelsCapable
from mila_llm_server.config import settings, loaded


class AnthropicModelsAdapter(ModelsCapable):

    @property
    def models_path(self) -> str:
        return "/v1/models"

    def format_models_response(self) -> dict:
        card = {
            "type": "model",
            "id": loaded.name,
            "display_name": loaded.name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "context_window": settings.context_length,
            "max_output_tokens": settings.default_max_new_tokens,
            # Beyond Anthropic's model object, which carries no lineage, and matching the
            # OpenAI card field for field so a client does not have to learn two spellings.
            "base_model": loaded.base_model,
            "license": loaded.license,
            "attribution": loaded.attribution,
        }

        return {
            "data": [card],
            "has_more": False,
            "first_id": loaded.name,
            "last_id": loaded.name,
        }
