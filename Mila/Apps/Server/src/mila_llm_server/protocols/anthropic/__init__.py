from mila_llm_server.protocols.base import ChatEndpoint, ProtocolAdapter

from .messages import AnthropicMessages
from .models import AnthropicModelsAdapter


class AnthropicAdapter(AnthropicModelsAdapter, ProtocolAdapter):
    """The Anthropic protocol: POST /v1/messages and GET /v1/models."""

    def chat_endpoints(self) -> list[ChatEndpoint]:
        return [AnthropicMessages()]
