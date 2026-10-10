"""
A reply says why it ended, and a streamed reply ends where the buffered one does.

A client reads the end reason to tell a finished answer from one cut off -- an agent that sees
`stop` on a truncated reply acts on half of it. And a stop sequence applies to a streamed reply
as it does to a buffered one, or the same request gives two different answers depending on how
it was asked. A fake worker stands in for the model, so the routes run with no store and no GPU.
"""
import dataclasses
import json
import random

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mila_llm_server.config import ModelFamily, loaded
from mila_llm_server.protocols.anthropic import AnthropicAdapter
from mila_llm_server.protocols.openai.chat import OpenAIChatAdapter
from mila_llm_server.protocols.openai.responses import OpenAIResponsesAdapter
from mila_llm_server.protocols.utils import (
    CONTEXT_LIMIT,
    END_TURN,
    MAX_TOKENS,
    StopScanner,
    anthropic_stop_reason,
    finish_reason_from_status,
    openai_finish_reason,
    truncate_at_stop,
)
from mila_llm_server.routes import factory
from mila_llm_server.schemas.internal import InferenceResponse


# ---------------------------------------------------------------------------
# The reason, from the binding's status
# ---------------------------------------------------------------------------

def test_a_budget_the_request_asked_for_is_max_tokens():
    assert finish_reason_from_status("length", requested_tokens=64, allowed_tokens=64) == MAX_TOKENS


def test_a_budget_the_context_lowered_is_the_context_limit():
    assert finish_reason_from_status("length", requested_tokens=4096, allowed_tokens=100) == CONTEXT_LIMIT
    assert finish_reason_from_status("context_limit", requested_tokens=64, allowed_tokens=64) == CONTEXT_LIMIT


@pytest.mark.parametrize("status", ["stop", "cancelled"])
def test_an_ended_turn_is_end_turn(status):
    assert finish_reason_from_status(status, requested_tokens=64, allowed_tokens=64) == END_TURN


def test_each_wire_spells_a_cut_reply_as_a_cut():
    assert openai_finish_reason(MAX_TOKENS) == "length"
    assert openai_finish_reason(CONTEXT_LIMIT) == "length"
    assert openai_finish_reason(END_TURN) == "stop"
    assert anthropic_stop_reason(MAX_TOKENS) == "max_tokens"
    assert anthropic_stop_reason(CONTEXT_LIMIT) == "model_context_window_exceeded"


# ---------------------------------------------------------------------------
# Stop sequences in a stream
# ---------------------------------------------------------------------------

def _scan(chunks, stop):
    scanner = StopScanner(stop)
    sent = "".join(scanner.feed(chunk) for chunk in chunks)

    return sent + (scanner.flush() if scanner.matched is None else ""), scanner.matched


def test_a_sequence_split_across_chunks_is_still_found():
    assert _scan(["The answer is 6\nQ", ": next"], ["Q:"]) == ("The answer is 6\n", "Q:")


def test_text_that_only_began_like_a_sequence_is_released():
    assert _scan(["Answer Q", "uickly"], ["Q:"]) == ("Answer Quickly", None)


def test_nothing_after_the_sequence_is_sent():
    scanner = StopScanner(["</s>"])

    assert scanner.feed("done</s> and more") == "done"
    assert scanner.feed(" even more") == ""


@pytest.mark.parametrize("seed", range(20))
def test_any_split_of_a_reply_ends_where_the_whole_reply_is_cut(seed):
    generator = random.Random(seed)
    text = "".join(generator.choice("ab Q:\n<") for _ in range(60))
    stop = ["Q:", "<|", "\n\n"]
    cuts = sorted(generator.sample(range(1, len(text)), 8))
    chunks = [text[start:end] for start, end in zip([0] + cuts, cuts + [len(text)])]

    assert _scan(chunks, stop) == truncate_at_stop(text, stop)


# ---------------------------------------------------------------------------
# The routes, over a fake worker
# ---------------------------------------------------------------------------

class FakeWorker:
    """A reply as chunks, and the status the binding reports after it."""

    def __init__(self, chunks, status):
        self.chunks = chunks
        self.status = status

    async def encode(self, text):
        return [128000, 1, 2, 3]

    async def decode(self, ids):
        return "".join(self.chunks)

    async def generate(self, prompt_ids, max_new_tokens, temperature, top_k, top_p=1.0):
        return list(prompt_ids) + list(range(len(self.chunks))), self.status

    async def generate_streaming(self, prompt_ids, on_text, max_new_tokens, temperature, top_k,
                                 top_p=1.0, stop_ctrl=None, strip_control_tokens=True):
        for chunk in self.chunks:
            if stop_ctrl is not None and stop_ctrl.stop_requested:
                return "cancelled"

            on_text(chunk)

        return self.status


@pytest.fixture
def serve(monkeypatch):
    before = dataclasses.replace(loaded)
    loaded.family = ModelFamily.llama
    loaded.context_length = 4096

    def start(adapter, chunks, status):
        monkeypatch.setattr(factory, "worker", FakeWorker(chunks, status))
        app = FastAPI()
        factory.register_routes(app, adapter)

        return TestClient(app)

    yield start

    for field in dataclasses.fields(before):
        setattr(loaded, field.name, getattr(before, field.name))


def _openai_stream(http, body):
    events = [line[6:] for line in http.post("/v1/completions", json={**body, "stream": True}).text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    chunks = [json.loads(event)["choices"][0] for event in events]

    return "".join(chunk["delta"]["content"] for chunk in chunks), chunks[-1]["finish_reason"]


CHUNKS = ["The final", " answer is 6\n", "Q", ": another question"]


def test_a_reply_cut_at_max_tokens_says_so_buffered_and_streamed(serve):
    http = serve(OpenAIChatAdapter(), CHUNKS, "length")
    body = {"prompt": [1, 2], "max_tokens": 4}

    buffered = http.post("/v1/completions", json=body).json()["choices"][0]

    assert buffered["finish_reason"] == "length"
    assert _openai_stream(http, body) == (buffered["text"], "length")


def test_buffered_and_streamed_replies_end_at_the_same_stop_sequence(serve):
    http = serve(OpenAIChatAdapter(), CHUNKS, "length")
    body = {"prompt": [1, 2], "max_tokens": 64, "stop": ["Q:"]}

    buffered = http.post("/v1/completions", json=body).json()["choices"][0]

    assert buffered["text"] == "The final answer is 6\n"
    assert buffered["finish_reason"] == "stop"
    assert _openai_stream(http, body) == (buffered["text"], "stop")


def test_a_reply_that_finished_says_stop(serve):
    http = serve(OpenAIChatAdapter(), CHUNKS, "stop")

    assert http.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 64}).json()["choices"][0]["finish_reason"] == "stop"


def _anthropic_stream_delta(http, body):
    events = [json.loads(line[6:]) for line in http.post("/v1/messages", json={**body, "stream": True}).text.splitlines()
              if line.startswith("data: ")]

    return [event["delta"] for event in events if event["type"] == "message_delta"][-1]


def test_anthropic_names_the_cut_and_the_sequence(serve):
    http = serve(AnthropicAdapter(), CHUNKS, "length")
    message = {"messages": [{"role": "user", "content": "Hi"}]}

    cut = http.post("/v1/messages", json={**message, "max_tokens": 4}).json()
    stopped = http.post("/v1/messages", json={**message, "max_tokens": 64, "stop_sequences": ["Q:"]}).json()

    assert cut["stop_reason"] == "max_tokens"
    assert _anthropic_stream_delta(http, {**message, "max_tokens": 4})["stop_reason"] == "max_tokens"
    assert (stopped["stop_reason"], stopped["stop_sequence"]) == ("stop_sequence", "Q:")
    assert _anthropic_stream_delta(http, {**message, "max_tokens": 64, "stop_sequences": ["Q:"]}) == {
        "stop_reason": "stop_sequence", "stop_sequence": "Q:"}


def test_a_cut_off_response_is_incomplete_and_a_finished_one_completed():
    adapter = OpenAIResponsesAdapter()

    cut = adapter.format_responses_response(InferenceResponse(text="The final", finish_reason=MAX_TOKENS))
    finished = adapter.format_responses_response(InferenceResponse(text="The final answer is 6"))
    streamed = json.loads(adapter.format_responses_stream_done("resp-1", "The final", CONTEXT_LIMIT).split("data: ", 1)[1])

    assert (cut["status"], cut["incomplete_details"]) == ("incomplete", {"reason": "max_output_tokens"})
    assert (finished["status"], finished["incomplete_details"]) == ("completed", None)
    assert (streamed["type"], streamed["response"]["status"]) == ("response.incomplete", "incomplete")
