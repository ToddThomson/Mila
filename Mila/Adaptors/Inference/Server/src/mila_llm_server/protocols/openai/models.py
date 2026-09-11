"""
OpenAI Models API protocol adapter.
  Models:  GET /v1/models
"""
import time

from mila_llm_server.protocols.base import ModelsCapable
from mila_llm_server.config import settings, loaded, ModelFamily

#: Families whose prompt assembly renders tool declarations and whose output is parsed for
#: a call. A client reads this card to decide how to drive the model -- Codex degrades its
#: whole loop when the card and the server disagree -- so claiming tools for a family that
#: has neither half wired produces calls nothing ever executes.
_TOOL_CAPABLE_FAMILIES = frozenset({
    ModelFamily.gemma,
    ModelFamily.llama,
    ModelFamily.qwen,
})


class OpenAIModelsAdapter(ModelsCapable):

    @property
    def models_path(self) -> str:
        return "/v1/models"

    def format_models_response(self) -> dict:
        created = int(time.time())
        tool_capable = loaded.family in _TOOL_CAPABLE_FAMILIES

        card = {
            "id": loaded.name,
            "object": "model",
            "created": created,
            "owned_by": "mila",
            "context_window": settings.context_length,
            "max_output_tokens": settings.default_max_new_tokens,
            "supports_parallel_tool_calls": False,
            "supports_reasoning": False,
            "reasoning_summary_format": "none",
            "slug": loaded.name,
            "display_name": loaded.name,
            "capabilities": {
                "tools": tool_capable,
                "apply_patch": tool_capable,
                "exec_command": tool_capable,
            },
            # Beyond OpenAI's model object, which carries no lineage. A client that lists
            # models is the one place a served model is presented to a person, so a license
            # requiring displayed attribution is answered here; attribution is empty when
            # the license asks for none. The card already carries non-OpenAI fields above,
            # so this is the established shape rather than a new one.
            "base_model": loaded.base_model,
            "license": loaded.license,
            "attribution": loaded.attribution,
        }

        return {
            "object": "list",
            "data": [card],
        }