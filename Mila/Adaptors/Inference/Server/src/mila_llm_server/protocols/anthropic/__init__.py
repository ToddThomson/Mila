from .messages import AnthropicMessagesAdapter
from .models import AnthropicModelsAdapter


class AnthropicAdapter(AnthropicMessagesAdapter, AnthropicModelsAdapter):
    """
    Unified Anthropic protocol adapter.
    Satisfies ProtocolAdapter      (POST /v1/messages)
              ModelsCapable        (GET  /v1/models)
    """
    pass
