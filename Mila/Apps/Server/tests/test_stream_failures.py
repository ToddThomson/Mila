"""
A stream the server fails -- a generation that stalls, or one that raises -- closes with the API's own error, never
as a normal end of turn: a client told the model finished cannot tell a server fault from the model choosing to stop.
"""
import asyncio
import dataclasses
import json

import mila
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mila_llm_server.config import loaded, settings
from mila_llm_server.protocols.anthropic import AnthropicAdapter
from mila_llm_server.protocols.openai import OpenAIAdapter
from mila_llm_server.routes import factory


class FailingWorker:
    """Writes a little reasoning, then either goes silent until stopped or raises."""

    def __init__(self, failure):
        self.failure = failure
        self.protocol = mila.ChatProtocol.gemma()

    async def encode(self, text):
        return [1, 2, 3]

    async def generate_streaming(self, prompt_ids, on_text, max_new_tokens, sampling, stop_ctrl=None,
                                 stop_at_markers=True, begins_in_reasoning=False):
        on_text("Let me think")

        if self.failure == "raise":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        while not stop_ctrl.stop_requested:
            await asyncio.sleep(0.02)

        return "cancelled"


@pytest.fixture
def serve(monkeypatch):
    before = dataclasses.replace(loaded)
    loaded.context_length = 4096
    monkeypatch.setattr(settings, "keepalive_interval", 0.05)
    monkeypatch.setattr(settings, "decode_timeout", 0.2)

    def start(adapter, failure):
        monkeypatch.setattr(factory, "worker", FailingWorker(failure))
        app = FastAPI()
        factory.register_routes(app, adapter)

        return TestClient(app)

    yield start

    for field in dataclasses.fields(before):
        setattr(loaded, field.name, getattr(before, field.name))


MESSAGES = [{"role": "user", "content": "Hi"}]


@pytest.mark.parametrize("failure", ["stall", "raise"])
def test_chat_completions_closes_with_an_error_chunk(serve, failure):
    http = serve(OpenAIAdapter(), failure)

    lines = http.post("/v1/chat/completions", json={"messages": MESSAGES, "stream": True}).text.splitlines()
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]

    assert events[-1]["error"]["type"] == "server_error"
    assert not any(choice.get("finish_reason") for event in events for choice in event.get("choices", []))


@pytest.mark.parametrize("failure", ["stall", "raise"])
def test_responses_closes_with_response_failed(serve, failure):
    http = serve(OpenAIAdapter(), failure)

    text = http.post("/v1/responses", json={"input": "Hi", "stream": True}).text
    names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]

    assert names[-1] == "response.failed"
    assert "response.completed" not in names and "response.incomplete" not in names


@pytest.mark.parametrize("failure", ["stall", "raise"])
def test_messages_closes_with_an_error_event(serve, failure):
    http = serve(AnthropicAdapter(), failure)

    text = http.post("/v1/messages", json={"messages": MESSAGES, "max_tokens": 64, "stream": True}).text
    names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]

    assert names[-1] == "error"
    assert "message_stop" not in names


def test_completions_closes_with_an_error_chunk(serve):
    http = serve(OpenAIAdapter(), "stall")

    lines = http.post("/v1/completions", json={"prompt": [1, 2], "stream": True}).text.splitlines()
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]

    assert events[-1]["error"]["type"] == "server_error"
