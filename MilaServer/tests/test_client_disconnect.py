"""
A client that leaves before its reply ends the generation, and the server says so: the work it abandoned is logged,
and the model stops rather than running on to the end of the context for nobody.
"""
import asyncio
import dataclasses
import json
import logging
import re
import time

import mila
import pytest
from fastapi import FastAPI

from mila_llm_server.config import loaded, settings
from mila_llm_server.model_worker import ModelWorker
from mila_llm_server.protocols.openai import OpenAIAdapter
from mila_llm_server.routes import factory
from mila_llm_server.schemas.internal import Sampling


class ThinkingWorker:
    """Reasons until it is stopped, as a model does that has not finished thinking when its client goes."""

    protocol = mila.ChatProtocol.gemma()
    failure = None

    def __init__(self):
        self.stops = []
        self.quiet = False

    async def encode(self, text):
        return [1, 2, 3]

    async def generate_streaming(self, prompt_ids, on_text, max_new_tokens, sampling, stop_ctrl=None,
                                 stop_at_markers=True, begins_in_reasoning=False):
        self.stops.append(stop_ctrl)
        started = time.monotonic()
        on_text("hmm ")

        while not stop_ctrl.stop_requested and time.monotonic() - started < 5:
            if not self.quiet:
                on_text("hmm ")

            await asyncio.sleep(0.01)

        return "cancelled"

    async def generate(self, prompt_ids, max_new_tokens, sampling, stop_ctrl=None):
        self.stops.append(stop_ctrl)
        started = time.monotonic()

        while not stop_ctrl.stop_requested and time.monotonic() - started < 5:
            await asyncio.sleep(0.01)

        return list(prompt_ids), "cancelled"


@pytest.fixture
def app(monkeypatch):
    before = dataclasses.replace(loaded)
    loaded.context_length = 4096
    monkeypatch.setattr(settings, "keepalive_interval", 0.05)
    monkeypatch.setattr(settings, "decode_timeout", 10.0)
    thinking = ThinkingWorker()
    monkeypatch.setattr(factory, "worker", thinking)
    served = FastAPI()
    factory.register_routes(served, OpenAIAdapter())

    yield served, thinking

    for field in dataclasses.fields(before):
        setattr(loaded, field.name, getattr(before, field.name))


async def leave_early(app, path, body, after_body_messages, linger=0.0):
    """
    Calls the app as uvicorn does (ASGI spec 2.3), and the client goes `linger` seconds after that many body messages
    are sent.
    """
    sent = []
    gone = asyncio.Event()
    request = [{"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}]

    async def receive():
        if request:
            if after_body_messages == 0:
                gone.set()

            return request.pop()

        # Once gone, uvicorn answers at once: Starlette's is_disconnected() reads a receive that waits as connected.
        await gone.wait()

        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

        if message["type"] == "http.response.body":
            if sum(1 for each in sent if each["type"] == "http.response.body") == after_body_messages:
                asyncio.get_running_loop().call_later(linger, gone.set)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=3)


MESSAGES = [{"role": "user", "content": "Hi"}]


def test_a_streamed_reply_whose_client_leaves_is_stopped_and_logged(app, caplog):
    served, thinking = app

    with caplog.at_level(logging.WARNING, logger="mila_llm_server.routes.factory"):
        asyncio.run(leave_early(served, "/v1/chat/completions", {"messages": MESSAGES, "stream": True}, 1, linger=0.2))

    assert thinking.stops[0].stop_requested
    assert re.search(r"client disconnected \d+ s into the reply, after [1-9]\d* chars of it", caplog.text)


@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"messages": MESSAGES, "stream": True}),
    ("/v1/completions", {"prompt": "Once", "max_tokens": 100, "stream": True}),
])
def test_a_streamed_reply_waiting_on_the_model_when_its_client_leaves_is_stopped_and_logged(app, caplog, monkeypatch,
                                                                                            path, body):
    # The stream is waiting for text, so only Starlette's cancellation can tell it the client went -- the path a long
    # silent stretch of the model takes, and the one that once left no record at all.
    served, thinking = app
    thinking.quiet = True
    monkeypatch.setattr(settings, "keepalive_interval", 5.0)

    with caplog.at_level(logging.WARNING, logger="mila_llm_server.routes.factory"):
        asyncio.run(leave_early(served, path, body, 1, linger=0.2))

    assert thinking.stops[0].stop_requested
    assert re.search(r"client disconnected \d+ s into the reply, after 4 chars of it", caplog.text)


def test_a_reply_not_streamed_whose_client_leaves_is_stopped_and_logged(app, caplog):
    served, thinking = app

    with caplog.at_level(logging.WARNING, logger="mila_llm_server.routes.factory"):
        asyncio.run(leave_early(served, "/v1/chat/completions", {"messages": MESSAGES}, 0))

    assert thinking.stops[0].stop_requested
    assert "client disconnected" in caplog.text


def test_a_completion_not_streamed_whose_client_leaves_is_stopped_and_logged(app, caplog):
    served, thinking = app

    with caplog.at_level(logging.WARNING, logger="mila_llm_server.routes.factory"):
        asyncio.run(leave_early(served, "/v1/completions", {"prompt": "Once", "max_tokens": 100}, 0))

    assert thinking.stops[0].stop_requested
    assert "client disconnected" in caplog.text


class EndlessModel:
    """Writes 'a' until stopped, on the worker thread, as the binding's generate does."""

    def generate(self, prompt_ids, on_token, max_new_tokens, stop_controller=None, **sampling):
        while not stop_controller.stop_requested:
            on_token(ord("a"))
            time.sleep(0.002)

        return "cancelled"


class ByteTokenizer:
    def decode(self, ids, errors="strict"):
        return bytes(ids).decode("utf-8", errors)


def test_a_generation_whose_caller_is_cancelled_still_logs_its_end(caplog):
    worker = ModelWorker()
    worker._model = EndlessModel()
    worker._tokenizer = ByteTokenizer()

    async def cancel_the_caller():
        stop_ctrl = mila.StopController()
        caller = asyncio.create_task(worker.generate_streaming([1], lambda text: None, 10_000, Sampling(), stop_ctrl))
        await asyncio.sleep(0.1)
        caller.cancel()
        stop_ctrl.request_stop()

        with pytest.raises(asyncio.CancelledError):
            await caller

    with caplog.at_level(logging.INFO, logger="mila_llm_server.model_worker"):
        asyncio.run(cancel_the_caller())
        worker._executor.shutdown(wait=True)

    assert re.search(r"generation ended: cancelled after [1-9]\d* tokens", caplog.text)
