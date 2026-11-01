"""
OpenAI Models API protocol adapter.
  Models:  GET /v1/models
"""
import time

from mila_llm_server.protocols.base import ModelsCapable
from mila_llm_server.config import loaded


class OpenAIModelsAdapter(ModelsCapable):

    @property
    def models_path(self) -> str:
        return "/v1/models"

    def format_models_response(self) -> dict:
        created = int(time.time())

        # What the manifest declares rather than what a family is assumed to do. A client reads this card to
        # decide how to drive the model -- Codex degrades its whole loop when the card and the server disagree.
        tool_capable = loaded.tools

        card = {
            "id": loaded.name,
            "object": "model",
            "created": created,
            "owned_by": "mila",
            "context_window": loaded.context_length,
            # The model has no output limit of its own, so one reply may fill the context.
            "max_output_tokens": loaded.context_length,
            "supports_parallel_tool_calls": False,
            "supports_reasoning": loaded.thinking,
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
            # What served it, for a client that records where a measurement came from. The PCI
            # address is written as nvidia-smi writes it, so the card can be found in either
            # tool's numbering.
            "mila_version": loaded.mila_version,
            "device": loaded.device,
        }

        return {
            "object": "list",
            "data": [card],
        }