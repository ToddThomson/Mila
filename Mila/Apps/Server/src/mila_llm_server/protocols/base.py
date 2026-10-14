"""
What each API implements for the route factory. A chat endpoint is a parser into the conversation and a
formatter from the reply, and knows nothing about which model is loaded (InferenceServerWire.md 3.3).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mila_llm_server.conversation import ServedReply, ServedRequest
from mila_llm_server.schemas.internal import InferenceRequest, InferenceResponse


@dataclass
class ReplyContext:
    """What a formatter needs besides the reply: the ids and names the wire carries."""

    response_id: str
    model: str
    created: int
    prompt_tokens: int

    # What the endpoint's own parse noted for its formatter -- Chat Completions' include_usage, Responses'
    # include -- and that nothing between the two reads.
    wire: dict = field(default_factory=dict)


class ChatEndpoint(ABC):
    """One conversational API: Chat Completions, Responses or Messages."""

    @property
    @abstractmethod
    def path(self) -> str: ...

    @abstractmethod
    def parse(self, body: dict, protocol) -> ServedRequest:
        """The request as a conversation. Raises RequestError for anything it refuses, naming it."""

    @abstractmethod
    def format(self, reply: ServedReply, context: ReplyContext) -> dict:
        """A whole reply in the API's response shape."""

    @abstractmethod
    def stream_open(self, context: ReplyContext) -> str:
        """The events that open a stream, sent before the prompt is read so the client holds the connection."""

    @abstractmethod
    def stream_keepalive(self, context: ReplyContext) -> str:
        """An event that keeps a client's read timeout from firing while the model works."""

    @abstractmethod
    def stream_reply(self, reply: ServedReply, context: ReplyContext) -> str:
        """Every remaining event of a stream, the reply's and the close."""

    def error(self, message: str) -> dict:
        return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}


class CompletionsCapable(ABC):
    """Raw text continuation, which has no conversation and so no protocol."""

    @property
    @abstractmethod
    def completions_path(self) -> str: ...

    @abstractmethod
    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]: ...

    @abstractmethod
    def format_completions_response(self, response: InferenceResponse) -> dict: ...

    @abstractmethod
    def format_stream_chunk(self, text: str, done: bool, finish_reason: str = "end_turn") -> str: ...

    @abstractmethod
    def format_stream_done(self) -> str: ...


class ModelsCapable(ABC):
    """GET /v1/models, registered only for an adapter that inherits it."""

    @property
    @abstractmethod
    def models_path(self) -> str: ...

    @abstractmethod
    def format_models_response(self) -> dict: ...


class ProtocolAdapter(ABC):
    """The API family a server speaks: the chat endpoints it serves, and optionally completions and models."""

    @abstractmethod
    def chat_endpoints(self) -> list[ChatEndpoint]: ...
