"""
A generation that raises takes the model down: the request gets the error, health reports the failure, later
requests are refused, and mila-server exits 1 -- never a model that goes on serving from a state nothing vouches for.
"""
import asyncio
import dataclasses

import mila
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mila_llm_server import app as server_app
from mila_llm_server.config import loaded
from mila_llm_server.model_worker import ModelWorker, worker
from mila_llm_server.protocols.openai import OpenAIAdapter
from mila_llm_server.routes import factory
from mila_llm_server.routes.health import router as health_router
from mila_llm_server.schemas.internal import Sampling

ILLEGAL_ACCESS = "CUDA error: an illegal memory access was encountered"


class FailingModel:
    """Writes two tokens, then fails as the binding does when a kernel faults."""

    def generate(self, prompt_ids, on_token, max_new_tokens, stop_controller=None, **sampling):
        on_token(ord("a"))
        on_token(ord("b"))

        raise RuntimeError(ILLEGAL_ACCESS)


class ByteTokenizer:
    def decode(self, ids, errors="strict"):
        return bytes(ids).decode("utf-8", errors)


@pytest.fixture
def failing_worker():
    failing = ModelWorker()
    failing._model = FailingModel()
    failing._tokenizer = ByteTokenizer()
    calls = []
    failing.on_failure = lambda: calls.append(True)

    yield failing, calls

    failing._executor.shutdown(wait=True)


@pytest.fixture
def healthy_again():
    yield

    worker.failure = None
    worker.on_failure = lambda: None


def test_a_streamed_generation_that_raises_takes_the_model_down(failing_worker):
    failing, calls = failing_worker

    with pytest.raises(RuntimeError, match="illegal memory access"):
        asyncio.run(failing.generate_streaming([1], lambda text: None, 100, Sampling(), mila.StopController()))

    assert failing.failure == f"RuntimeError: {ILLEGAL_ACCESS}"
    assert calls == [True]


def test_a_generation_that_raises_takes_the_model_down_once(failing_worker):
    failing, calls = failing_worker

    for _ in range(2):
        with pytest.raises(RuntimeError):
            asyncio.run(failing.generate([1], 100, Sampling(), mila.StopController()))

    assert failing.failure == f"RuntimeError: {ILLEGAL_ACCESS}"
    assert calls == [True]


def test_health_reports_the_failure(healthy_again):
    app = FastAPI()
    app.include_router(health_router)
    http = TestClient(app)

    assert http.get("/v1/health").json() == {"status": "ok"}

    worker.failure = f"RuntimeError: {ILLEGAL_ACCESS}"

    response = http.get("/v1/health")
    assert response.status_code == 503
    assert response.json() == {"status": "failed", "error": worker.failure}
    assert http.get("/").status_code == 503


class DownWorker:
    failure = f"RuntimeError: {ILLEGAL_ACCESS}"
    protocol = mila.ChatProtocol.gemma()

    async def encode(self, text):
        raise AssertionError("a model that is down is not asked to encode")


@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}),
    ("/v1/completions", {"prompt": "Once", "max_tokens": 4}),
])
def test_requests_after_the_failure_are_refused(monkeypatch, path, body):
    before = dataclasses.replace(loaded)
    loaded.context_length = 4096
    monkeypatch.setattr(factory, "worker", DownWorker())
    app = FastAPI()
    factory.register_routes(app, OpenAIAdapter())

    try:
        response = TestClient(app).post(path, json=body)
    finally:
        for field in dataclasses.fields(before):
            setattr(loaded, field.name, getattr(before, field.name))

    assert response.status_code == 503
    assert ILLEGAL_ACCESS in response.json()["error"]["message"]


def test_mila_server_stops_and_exits_1_when_the_model_fails(monkeypatch, healthy_again):
    servers = []

    class ServerThatFails:
        def __init__(self, config):
            self.should_exit = False
            servers.append(self)

        def run(self):
            worker._model_failed(RuntimeError(ILLEGAL_ACCESS))

    monkeypatch.setattr(server_app.uvicorn, "Server", ServerThatFails)

    with pytest.raises(SystemExit) as stopped:
        server_app.serve()

    assert stopped.value.code == 1
    assert servers[0].should_exit
