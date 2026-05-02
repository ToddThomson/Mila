"""
Abstract base that every protocol adapter must implement. The route factory
depends only on this interface.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator

from fastapi import Request
from fastapi.responses import Response

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
        prompt_ids on the returned request will be empty — the factory fills
        them after encoding.
        Returns the prompt string so the factory can encode it via the worker.
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
