"""
The Completions route takes OpenAI's token-id prompts and honours `stop` on a buffered reply.

An evaluation harness depends on both. It sends the ids its own tokenizer produced, so that
every engine it compares sees the same input, and it leaves stop sequences to the server --
a reply that runs past one scores differently from the same reply cut at it. A fake worker
stands in for the model, so the route runs with no store and no GPU.
"""
import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mila

from mila_llm_server.config import loaded
from mila_llm_server.protocols.openai import OpenAIAdapter
from mila_llm_server.protocols.openai.chat import OpenAIChatCompletions, OpenAICompletions
from mila_llm_server.protocols.utils import (
    CONTEXT_LIMIT, MAX_TOKENS, finish_reason_from_status, parse_completion_prompt, parse_stop, truncate_at_stop)
from mila_llm_server.routes import factory


# ---------------------------------------------------------------------------
# The parsing helpers
# ---------------------------------------------------------------------------

def test_stop_accepts_a_string_a_list_or_nothing():
    assert parse_stop(None) == []
    assert parse_stop("Q:") == ["Q:"]
    assert parse_stop(["Q:", "", "</s>"]) == ["Q:", "</s>"]


def test_truncation_cuts_at_the_earliest_stop_whatever_its_order_in_the_list():
    text, matched = truncate_at_stop("The answer is 4.\nQ: next\n</s>", ["</s>", "Q:"])

    assert text == "The answer is 4.\n"
    assert matched == "Q:"


def test_truncation_without_a_match_returns_the_text_whole():
    assert truncate_at_stop("The answer is 4.", ["Q:"]) == ("The answer is 4.", None)
    assert truncate_at_stop("The answer is 4.", []) == ("The answer is 4.", None)


@pytest.mark.parametrize("prompt", [[128000, 9906, 1917], [[128000, 9906, 1917]]])
def test_token_prompts_are_read_as_ids(prompt):
    assert parse_completion_prompt(prompt) == ("", [128000, 9906, 1917])


@pytest.mark.parametrize("prompt", ["Hello", ["Hello"]])
def test_text_prompts_are_read_as_text(prompt):
    assert parse_completion_prompt(prompt) == ("Hello", [])


@pytest.mark.parametrize("prompt", [[], ["one", "two"], [[1, 2], [3, 4]], 7])
def test_batches_and_malformed_prompts_are_refused(prompt):
    with pytest.raises(ValueError):
        parse_completion_prompt(prompt)


def test_the_openai_adapter_carries_ids_and_stop_into_the_request():
    adapter = OpenAICompletions()

    prompt_str, request = adapter.parse_completions_request(
        {"prompt": [[128000, 9906]], "stop": ["Q:", "<|eot_id|>"], "max_tokens": 32, "temperature": 0}
    )

    assert prompt_str == ""
    assert request.prompt_ids == [128000, 9906]
    assert request.stop == ["Q:", "<|eot_id|>"]


def test_a_chat_request_carries_stop_too():
    request = OpenAIChatCompletions().parse(
        {"messages": [{"role": "user", "content": "Hi"}], "stop": "Q:"}, mila.ChatProtocol.gemma())

    assert request.stop == ["Q:"]


# ---------------------------------------------------------------------------
# The route, over a fake worker
# ---------------------------------------------------------------------------

class FakeWorker:
    """Records what the route sends and answers with a fixed reply."""

    failure = None

    def __init__(self, reply_ids):
        self.reply_ids = reply_ids
        self.encoded = []
        self.generated_from = None
        self.sampling = None
        self.length = None

    async def encode(self, text):
        self.encoded.append(text)

        return [128000, 1, 2, 3]

    async def decode(self, ids):
        return "The final answer is 6\nQ: another question"

    async def generate(self, prompt_ids, max_new_tokens, sampling, stop_ctrl=None):
        self.generated_from = list(prompt_ids)
        self.sampling = sampling
        self.length = max_new_tokens

        return list(prompt_ids) + self.reply_ids, "stop"


@pytest.fixture
def client(monkeypatch):
    before = dataclasses.replace(loaded)
    loaded.context_length = 4096

    fake = FakeWorker(reply_ids=[10, 11, 12])
    monkeypatch.setattr(factory, "worker", fake)

    app = FastAPI()
    factory.register_routes(app, OpenAIAdapter())

    yield TestClient(app), fake

    for field in dataclasses.fields(before):
        setattr(loaded, field.name, getattr(before, field.name))


def test_token_ids_reach_the_model_unencoded(client):
    http, fake = client

    response = http.post("/v1/completions", json={"prompt": [[128000, 9906, 1917]], "max_tokens": 8, "temperature": 0})

    assert response.status_code == 200
    assert fake.encoded == []
    assert fake.generated_from == [128000, 9906, 1917]
    assert response.json()["usage"]["prompt_tokens"] == 3


def test_a_text_prompt_is_still_encoded(client):
    http, fake = client

    response = http.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 8})

    assert response.status_code == 200
    assert fake.encoded == ["Hello"]


def test_the_reply_is_cut_at_the_stop_sequence(client):
    http, _ = client

    response = http.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 8, "stop": ["Q:", "<|eot_id|>"]})

    assert response.json()["choices"][0]["text"] == "The final answer is 6\n"


def test_without_stop_the_reply_is_whole(client):
    http, _ = client

    response = http.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 8})

    assert response.json()["choices"][0]["text"] == "The final answer is 6\nQ: another question"


def test_a_completion_naming_no_length_may_fill_the_context(client):
    http, fake = client

    http.post("/v1/completions", json={"prompt": [1, 2, 3]})

    assert fake.length == 4096 - 3


def test_running_out_with_no_length_named_is_the_context_ending():
    assert finish_reason_from_status("length", None, 500) == CONTEXT_LIMIT
    assert finish_reason_from_status("length", 500, 500) == MAX_TOKENS
    assert finish_reason_from_status("length", 900, 500) == CONTEXT_LIMIT


def test_the_seed_reaches_the_model_and_a_malformed_one_is_refused(client):
    http, fake = client

    assert http.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 8, "seed": -2}).status_code == 200
    assert fake.sampling.seed == 2**64 - 2

    fake.sampling = None
    refused = http.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 8, "seed": "7"})

    assert refused.status_code == 400 and "seed" in refused.text
    assert fake.sampling is None


def test_a_batch_is_refused_as_an_invalid_request(client):
    http, fake = client

    response = http.post("/v1/completions", json={"prompt": [[1, 2], [3, 4]], "max_tokens": 8})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert fake.generated_from is None
