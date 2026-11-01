from mila_llm_server.protocols.base import ChatEndpoint, ProtocolAdapter

from .chat import OpenAIChatCompletions, OpenAICompletions
from .models import OpenAIModelsAdapter
from .responses import OpenAIResponses


class OpenAIAdapter(OpenAICompletions, OpenAIModelsAdapter, ProtocolAdapter):
    """
    The OpenAI protocol: POST /v1/chat/completions and POST /v1/responses, POST /v1/completions, and
    GET /v1/models.
    """

    def chat_endpoints(self) -> list[ChatEndpoint]:
        return [OpenAIChatCompletions(), OpenAIResponses()]
