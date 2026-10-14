"""
A tool call reaches a Responses client as a complete function_call item, for every family.

The Gemma and Qwen bridges are shared with the Anthropic adapter and return only
{'call_id', 'name', 'arguments'}. The Responses wire also needs the item's 'type' and 'id', and
when the bridges were folded into the runtime nothing added them back: every Gemma tool call
over /v1/responses died with KeyError: 'id' mid-stream, which Codex reports as a transport error.
These drive raw model text through the adapter exactly as the streaming and buffered routes do.
"""
import dataclasses
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mila_llm_server.config import loaded, ModelFamily
from mila_llm_server.protocols.openai.responses import OpenAIResponsesAdapter
from mila_llm_server.routes import factory
from mila_llm_server.schemas.internal import InferenceResponse


GEMMA_CALL = '<|tool_call>call:exec_command{cmd:<|"|>cat line_count.txt<|"|>}<tool_call|>'

QWEN_CALL = (
    "<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat line_count.txt\n</parameter>\n"
    "</function>\n</tool_call>"
)


@pytest.fixture(params=[(ModelFamily.gemma, GEMMA_CALL), (ModelFamily.qwen, QWEN_CALL)],
                ids=["gemma", "qwen"])
def raw_call(request, monkeypatch):
    family, text = request.param
    monkeypatch.setattr(loaded, "family", family)

    return text


def test_parsed_call_is_a_complete_function_call_item(raw_call):
    item = OpenAIResponsesAdapter().parse_tool_call_from_text(raw_call)

    assert item["type"] == "function_call"
    assert item["id"].startswith("fc_")
    assert item["call_id"].startswith("call_")
    assert item["name"] == "exec_command"
    assert json.loads(item["arguments"]) == {"cmd": "cat line_count.txt"}


def test_streamed_call_formats_all_four_events(raw_call):
    adapter = OpenAIResponsesAdapter()
    item = adapter.parse_tool_call_from_text(raw_call)

    stream = adapter.format_responses_stream_function_call("resp-test", item)
    stream += adapter.format_responses_stream_done_with_tool_call("resp-test", item)

    events = [line.removeprefix("event: ") for line in stream.splitlines() if line.startswith("event: ")]

    assert events == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]


def test_buffered_response_carries_the_call_as_its_output(raw_call):
    body = OpenAIResponsesAdapter().format_responses_response(InferenceResponse(text=raw_call))

    assert [item["type"] for item in body["output"]] == ["function_call"]
    assert body["output"][0]["id"].startswith("fc_")


class StrippingWorker:
    """
    The worker's two buffered paths as they differ: the raw collect keeps the call's markers,
    decode() strips them, as the binding's Gemma decode does.
    """

    def __init__(self, raw):
        self.raw = raw

    async def encode(self, text):
        return [2, 1, 2, 3]

    async def generate(self, prompt_ids, max_new_tokens, temperature, top_k, top_p=1.0):
        return list(prompt_ids) + [7], "stop"

    async def decode(self, ids):
        return self.raw.replace("<|tool_call>", "").replace("<tool_call|>", "").replace('<|"|>', "")

    async def generate_collect(self, prompt_ids, max_new_tokens, temperature, top_k, top_p=1.0):
        return self.raw, 1, "stop"


def test_buffered_route_returns_the_call_the_model_made(raw_call, monkeypatch):
    # A buffered Gemma call over /v1/responses came back as stripped text, repeated to the budget.
    before = dataclasses.replace(loaded)
    loaded.context_length = 4096
    monkeypatch.setattr(factory, "worker", StrippingWorker(raw_call))
    app = FastAPI()
    factory.register_routes(app, OpenAIResponsesAdapter())

    try:
        body = TestClient(app).post("/v1/responses", json={"input": "What does line_count.txt hold?"}).json()
    finally:
        for field in dataclasses.fields(before):
            setattr(loaded, field.name, getattr(before, field.name))

    assert [item["type"] for item in body["output"]] == ["function_call"]
    assert body["output"][0]["name"] == "exec_command"
