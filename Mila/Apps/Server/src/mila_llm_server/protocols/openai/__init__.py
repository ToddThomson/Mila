from .chat import OpenAIChatAdapter
from .responses import OpenAIResponsesAdapter
from .models import OpenAIModelsAdapter


class OpenAIAdapter(OpenAIChatAdapter, OpenAIResponsesAdapter, OpenAIModelsAdapter):
    """
    Unified OpenAI protocol adapter.
    Satisfies ProtocolAdapter      (POST /v1/chat/completions, POST /v1/completions)
              ResponsesCapable     (POST /v1/responses)
              ModelsCapable        (GET  /v1/models)
    """
    pass