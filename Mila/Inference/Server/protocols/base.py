"""
Abstract base that every protocol adapter must implement. The route factory
depends only on this interface.
"""
from abc import ABC, abstractmethod

from schemas.internal import InferenceRequest, InferenceResponse


class ProtocolAdapter(ABC):

    @property
    @abstractmethod
    def chat_path(self) -> str:
        """URL path for the chat/messages endpoint."""
        ...

    @property
    @abstractmethod
    def completions_path(self) -> str:
        """URL path for the raw completions endpoint."""
        ...

    @abstractmethod
    def parse_chat_request(self, body: dict) -> tuple[str, InferenceRequest]:
        """
        Parse a raw request body into a (raw_prompt_str, InferenceRequest).
        prompt_ids on the returned request will be empty; the factory fills
        them after encoding. Returns the prompt string so the factory can
        encode it via the worker.
        """
        ...

    @abstractmethod
    def parse_completions_request(self, body: dict) -> tuple[str, InferenceRequest]:
        """Same contract as parse_chat_request but for /completions."""
        ...

    @abstractmethod
    def format_chat_response(self, response: InferenceResponse) -> dict:
        """Serialize a completed InferenceResponse into the protocol's JSON shape."""
        ...

    @abstractmethod
    def format_completions_response(self, response: InferenceResponse) -> dict:
        """Serialize a completed InferenceResponse into the completions JSON shape."""
        ...

    @abstractmethod
    def format_stream_chunk(self, text: str, done: bool) -> str:
        """
        Format a single token (or the done sentinel) into an SSE line.
        Must return a complete SSE-formatted string including trailing newlines.
        """
        ...

    @abstractmethod
    def format_stream_done(self) -> str:
        """Final SSE line(s) sent when the stream closes cleanly."""
        ...


class ResponsesCapable(ABC):
    """
    Optional mixin for adapters that support the OpenAI Responses API
    (POST /v1/responses). The route factory registers the /v1/responses
    endpoint only when the adapter inherits from this class.
    """

    @property
    @abstractmethod
    def responses_path(self) -> str:
        """URL path for the responses endpoint."""
        ...

    @abstractmethod
    def parse_responses_request(self, body: dict) -> tuple[str, InferenceRequest]:
        """Same contract as parse_chat_request but for /responses."""
        ...

    @abstractmethod
    def format_responses_response(self, response: InferenceResponse) -> dict:
        """Serialize a completed InferenceResponse into the responses JSON shape."""
        ...

    @abstractmethod
    def format_responses_stream_created(self, response_id: str) -> str:
        """
        First SSE event emitted when a Responses API stream opens, before prefill begins.
        Must be sent immediately so the client holds the connection open during prefill.
        """
        ...

    @abstractmethod
    def format_responses_stream_keepalive(self, response_id: str) -> str:
        """
        Heartbeat SSE event emitted during prefill while waiting for the first token.
        Must carry a real data: payload — SSE comment lines (: ...) are ignored by
        most clients and do not reset their read timeout.
        """
        ...

    @abstractmethod
    def format_responses_stream_output_item_added(self, response_id: str, item_id: str) -> str: ...

    @abstractmethod
    def format_responses_stream_content_part_added(self, response_id: str) -> str: ...

    @abstractmethod
    def format_responses_stream_content_part_done(self, response_id: str, text: str) -> str: ...

    @abstractmethod
    def format_responses_stream_output_item_done(self, response_id: str, item_id: str, text: str) -> str: ...
    
    @abstractmethod
    def format_responses_stream_chunk(self, text: str, done: bool, response_id: str) -> str:
        """
        Format a single token (or done sentinel) into a Responses API SSE event.
        Emits response.output_text.delta while streaming and response.output_text.done
        on the final chunk. Must return a complete SSE-formatted string including
        trailing newlines.
        """
        ...

    @abstractmethod
    def format_responses_stream_done(self, response_id: str, output_text: str = "") -> str:
        """
        Final response.completed SSE event for the Responses API stream.
        Must include the full response object with status=completed and final output content.
        Must return a complete SSE-formatted string including trailing newlines.
        """
        ...
