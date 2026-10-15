"""
The three chat endpoints over a fake worker that holds a real protocol (InferenceServerWire.md 6).

Each API's request becomes one conversation, and each reply comes back in that API's own shape: every call the
model made, its reasoning, and why it ended. The prompt the worker is handed is the runtime protocol's rendering of
that conversation, so a wrong parse shows as a wrong prompt. No store, no GPU.
"""
import dataclasses
import json
import pathlib
import re

import mila
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mila_llm_server.config import loaded
from mila_llm_server.protocols.anthropic import AnthropicAdapter
from mila_llm_server.protocols.openai import OpenAIAdapter
from mila_llm_server.routes import factory

QWEN_TWO_CALLS = (
    "The user wants both.\n</think>\n\n"
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=get_time>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>")
# Thinking is on by default in Qwen 3.8's template, so its reply begins inside the channel.
QWEN_ANSWER = "The tool says sunny.\n</think>\n\nIt is sunny."
GEMMA_CALL = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
GEMMA_MALFORMED = '<|tool_call>call:exec_command{cmd="ls"}<tool_call|>'

WEATHER = {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}


class FakeWorker:
    def __init__(self, protocol, raw):
        self.protocol = protocol
        self.raw = raw
        self.prompts = []

    async def encode(self, text):
        self.prompts.append(text)

        return [1, 2, 3, 4]

    async def generate_streaming(self, prompt_ids, on_text, max_new_tokens, temperature, top_k,
                                 top_p=1.0, stop_ctrl=None, stop_at_markers=True):
        for piece in re.split(r"(?<=\n)", self.raw):
            on_text(piece)

        return "stop"


@pytest.fixture
def serve(monkeypatch):
    before = dataclasses.replace(loaded)
    loaded.context_length = 8192

    def start(adapter, protocol, raw):
        fake = FakeWorker(protocol, raw)
        monkeypatch.setattr(factory, "worker", fake)
        app = FastAPI()
        factory.register_routes(app, adapter)

        return TestClient(app), fake

    yield start

    for field in dataclasses.fields(before):
        setattr(loaded, field.name, getattr(before, field.name))


def _events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------

CHAT_RESUME = {
    "messages": [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
    ],
    "tools": [{"type": "function", "function": WEATHER}],
}


def test_chat_completions_returns_every_call_and_the_reasoning(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_TWO_CALLS)
    body = {"messages": [{"role": "user", "content": "Weather and time?"}], "tools": [{"type": "function", "function": WEATHER}]}

    choice = http.post("/v1/chat/completions", json=body).json()["choices"][0]

    assert choice["finish_reason"] == "tool_calls"
    assert [call["function"]["name"] for call in choice["message"]["tool_calls"]] == ["get_weather", "get_time"]
    assert choice["message"]["reasoning_content"] == "The user wants both."
    assert choice["message"]["content"] is None


def test_chat_completions_streams_the_same_calls_under_one_id(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_TWO_CALLS)
    body = {"messages": [{"role": "user", "content": "Weather and time?"}], "stream": True,
            "stream_options": {"include_usage": True}}

    chunks = _events(http.post("/v1/chat/completions", json=body).text)
    calls = [delta for chunk in chunks for choice in chunk["choices"] for delta in choice["delta"].get("tool_calls", [])]

    assert len({chunk["id"] for chunk in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert [call["function"]["name"] for call in calls] == ["get_weather", "get_time"]
    assert chunks[-1]["usage"]["completion_tokens"] > 0


def test_chat_completions_renders_the_result_as_a_tool_turn(serve):
    # N1: rendered as a user turn, Qwen reads the result as the person's message.
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)

    reply = http.post("/v1/chat/completions", json=CHAT_RESUME).json()

    assert reply["choices"][0]["message"]["content"] == "It is sunny."
    assert "<|im_start|>user\n<tool_response>\nSunny\n</tool_response><|im_end|>" in fake.prompts[0]
    assert "Be brief." in fake.prompts[0]


def test_parallel_tool_calls_false_keeps_the_first(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_TWO_CALLS)
    body = {"messages": [{"role": "user", "content": "Both?"}], "parallel_tool_calls": False}

    calls = http.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]["tool_calls"]

    assert [call["function"]["name"] for call in calls] == ["get_weather"]


def test_no_system_turn_is_invented(serve):
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), "Hi.")

    http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}]})

    assert "helpful assistant" not in fake.prompts[0]
    assert not fake.prompts[0].startswith("<bos><|turn>system")


@pytest.mark.parametrize("field, value", [
    ("n", 2), ("seed", 7), ("logprobs", True), ("frequency_penalty", 0.5), ("logit_bias", {"1": 2}),
    ("response_format", {"type": "json_object"}), ("tool_choice", "required"), ("reasoning_effort", "high"),
])
def test_a_field_that_cannot_be_acted_on_is_refused_by_name(serve, field, value):
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), "Hi.")

    response = http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}], field: value})

    assert response.status_code == 400
    assert field.split("_")[0] in response.json()["error"]["message"]
    assert fake.prompts == []


def test_a_field_at_its_default_is_accepted(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), "Hi.")
    body = {"messages": [{"role": "user", "content": "Hi"}], "n": 1, "frequency_penalty": 0, "user": "u1"}

    assert http.post("/v1/chat/completions", json=body).status_code == 200


def test_reasoning_effort_sets_the_templates_own_variable(serve):
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), "Hi.")

    http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "low"})

    assert "Reasoning effort is set to low." in fake.prompts[0]


@pytest.mark.parametrize("effort", ["low", "high"])
def test_a_level_turns_thinking_on_for_a_template_with_no_levels(serve, effort):
    # Gemma has one depth: any level asks for reasoning, and reasoning is all it can give.
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), "Hi.")

    response = http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": effort})

    assert response.status_code == 200
    assert fake.prompts[0].startswith("<bos><|turn>system\n<|think|>")


@pytest.mark.parametrize("protocol, ending", [
    (mila.ChatProtocol.gemma, "<|channel>thought\n<channel|>"), (mila.ChatProtocol.qwen, "<think>\n\n</think>\n\n")])
def test_reasoning_effort_none_turns_thinking_off(serve, protocol, ending):
    http, fake = serve(OpenAIAdapter(), protocol(), "Hi.")

    http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "none"})

    assert fake.prompts[0].endswith(ending)


def test_a_template_that_cannot_reason_takes_none_and_refuses_a_level(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.llama31(), "Hi.")
    body = {"messages": [{"role": "user", "content": "Hi"}]}

    assert http.post("/v1/chat/completions", json={**body, "reasoning_effort": "none"}).status_code == 200
    refused = http.post("/v1/chat/completions", json={**body, "reasoning_effort": "low"})
    assert refused.status_code == 400 and "does not reason" in refused.text


def test_responses_reasoning_effort_turns_gemmas_thinking_on(serve):
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), "Hi.")

    assert http.post("/v1/responses", json={"input": "Hi", "reasoning": {"effort": "medium"}}).status_code == 200
    assert fake.prompts[0].startswith("<bos><|turn>system\n<|think|>")


def test_a_result_that_answers_no_call_is_refused(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), "Hi.")
    body = {"messages": [{"role": "user", "content": "Hi"}, {"role": "tool", "tool_call_id": "nope", "content": "x"}]}

    response = http.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert "nope" in response.json()["error"]["message"]


def test_an_image_part_is_refused_not_dropped(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), "Hi.")
    content = [{"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": "x"}}]

    assert http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": content}]}).status_code == 400


def test_a_malformed_call_is_answered_as_text_not_dispatched(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), GEMMA_MALFORMED)

    message = http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "ls"}]}).json()["choices"][0]["message"]

    assert "tool_calls" not in message
    assert 'call:exec_command{cmd="ls"}' in message["content"]


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

RESPONSES_RESUME = {
    "instructions": "Be brief.",
    "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather in Paris?"}]},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Need the tool."}]},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "Sunny"},
    ],
    "tools": [{"type": "function", **WEATHER}, {"type": "web_search"}],
}


def test_responses_resumes_the_model_turn_after_a_result(serve):
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), "It is sunny.")

    output = http.post("/v1/responses", json=RESPONSES_RESUME).json()["output"]

    assert fake.prompts[0].endswith('response:get_weather{value:<|"|>Sunny<|"|>}<tool_response|>')
    assert output[-1]["content"][0]["text"] == "It is sunny."


def test_responses_returns_reasoning_then_every_call(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_TWO_CALLS)

    output = http.post("/v1/responses", json={"input": "Both?", "include": ["reasoning.encrypted_content"]}).json()["output"]

    assert [item["type"] for item in output] == ["reasoning", "function_call", "function_call"]
    assert output[0]["summary"][0]["text"] == "The user wants both."
    assert output[0]["encrypted_content"] == "The user wants both."
    assert output[1]["call_id"] != output[2]["call_id"]


def test_responses_streams_each_item_in_order(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_TWO_CALLS)

    events = _events(http.post("/v1/responses", json={"input": "Both?", "stream": True}).text)
    done = [event["item"]["type"] for event in events if event["type"] == "response.output_item.done"]

    assert events[0]["type"] == "response.created"
    assert done == ["reasoning", "function_call", "function_call"]
    assert events[-1]["type"] == "response.completed"
    assert [item["type"] for item in events[-1]["response"]["output"]] == done


def test_a_call_id_is_the_same_each_time_the_transcript_is_sent(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), GEMMA_CALL)

    first = http.post("/v1/responses", json={"input": "Weather?"}).json()["output"][0]["call_id"]
    second = http.post("/v1/responses", json={"input": "Weather?"}).json()["output"][0]["call_id"]

    assert first == second


@pytest.mark.parametrize("body", [
    {"previous_response_id": "resp_1"}, {"store": True}, {"text": {"format": {"type": "json_schema"}}},
    {"input": [{"type": "computer_call"}]}, {"include": ["message.input_image.image_url"]},
])
def test_responses_refuses_what_it_cannot_do(serve, body):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), "Hi.")

    assert http.post("/v1/responses", json={"input": "Hi", **body}).status_code == 400


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def test_messages_splits_a_tool_result_out_of_its_user_message(serve):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    body = {"max_tokens": 64, "system": "Be brief.", "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Need the tool.", "signature": "s"},
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"}]},
    ]}

    reply = http.post("/v1/messages", json=body).json()

    assert "<tool_response>\nSunny\n</tool_response>" in fake.prompts[0]
    # The thinking Claude Code sent back is the turn's reasoning, which Qwen keeps.
    assert "<think>\nNeed the tool.\n</think>" in fake.prompts[0]
    assert reply["content"] == [
        {"type": "thinking", "thinking": "The tool says sunny.", "signature": ""}, {"type": "text", "text": "It is sunny."}]


def test_messages_returns_thinking_then_every_tool_use(serve):
    http, _ = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_TWO_CALLS)

    reply = http.post("/v1/messages", json={"max_tokens": 64, "messages": [{"role": "user", "content": "Both?"}]}).json()

    assert [block["type"] for block in reply["content"]] == ["thinking", "tool_use", "tool_use"]
    assert reply["stop_reason"] == "tool_use"


def test_messages_streams_each_block_with_its_own_index(serve):
    http, _ = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_TWO_CALLS)
    body = {"max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "Both?"}]}

    events = _events(http.post("/v1/messages", json=body).text)
    starts = [(event["index"], event["content_block"]["type"]) for event in events if event["type"] == "content_block_start"]

    assert starts == [(0, "thinking"), (1, "tool_use"), (2, "tool_use")]
    assert [event for event in events if event["type"] == "message_delta"][0]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("extra", [
    {"thinking": {"type": "enabled", "budget_tokens": 1024}}, {"thinking": {"type": "adaptive", "display": "full"}},
    {"tool_choice": {"type": "any"}},
    {"output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}},
    {"output_config": {"task_budget": 1000}},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
])
def test_messages_refuses_what_it_cannot_do(serve, extra):
    http, _ = serve(AnthropicAdapter(), mila.ChatProtocol.gemma(), "Hi.")
    body = {"max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}], **extra}

    assert http.post("/v1/messages", json=body).status_code == 400


def test_thinking_disabled_sets_the_templates_switch(serve):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), "Hi.")
    body = {"max_tokens": 64, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "Hi"}]}

    http.post("/v1/messages", json=body)

    assert fake.prompts[0].endswith("<think>\n\n</think>\n\n")


def test_adaptive_thinking_turns_the_templates_thinking_on(serve):
    # Claude Code sends adaptive on every request; Gemma's template defaults to thinking off.
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.gemma(), "Hi.")
    body = {"max_tokens": 64, "thinking": {"type": "adaptive"}, "messages": [{"role": "user", "content": "Hi"}]}

    assert http.post("/v1/messages", json=body).status_code == 200
    assert fake.prompts[0].startswith("<bos><|turn>system\n<|think|>")
    assert not fake.prompts[0].endswith("<|channel>thought\n<channel|>")


@pytest.mark.parametrize("kind", ["adaptive", "disabled"])
def test_a_template_without_a_thinking_switch_meets_both_as_sent(serve, kind):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.llama31(), "Hi.")
    plain = {"max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}]}

    assert http.post("/v1/messages", json={**plain, "thinking": {"type": kind}}).status_code == 200
    http.post("/v1/messages", json=plain)
    assert fake.prompts[0] == fake.prompts[1]


def test_effort_sets_the_templates_own_variable(serve):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    plain = {"max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}]}

    assert http.post("/v1/messages", json={**plain, "output_config": {"effort": "low"}}).status_code == 200
    http.post("/v1/messages", json=plain)

    assert "Reasoning effort is set to low" in fake.prompts[0]
    assert "Reasoning effort is set to low" not in fake.prompts[1]


@pytest.mark.parametrize("protocol, effort", [(mila.ChatProtocol.gemma, "low"), (mila.ChatProtocol.qwen, "high")])
def test_an_effort_the_template_cannot_take_is_refused_by_name(serve, protocol, effort):
    http, fake = serve(AnthropicAdapter(), protocol(), "Hi.")
    body = {"max_tokens": 64, "output_config": {"effort": effort}, "messages": [{"role": "user", "content": "Hi"}]}

    response = http.post("/v1/messages", json=body)

    assert response.status_code == 400
    assert "output_config.effort" in response.text
    assert fake.prompts == []


EARLIER_CALL = {"max_tokens": 64, "messages": [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Plan the call.", "signature": ""},
        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"}]},
    {"role": "assistant", "content": "It is sunny."},
    {"role": "user", "content": "And tomorrow?"},
]}
KEEP_ALL = {"context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}}


def test_keeping_all_thinking_sets_the_templates_preserve_thinking(serve):
    # Gemma's default drops an earlier call's reasoning; preserve_thinking keeps it.
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.gemma(), "Hi.")

    assert http.post("/v1/messages", json={**EARLIER_CALL, **KEEP_ALL}).status_code == 200
    http.post("/v1/messages", json=EARLIER_CALL)

    assert "Plan the call." in fake.prompts[0]
    assert "Plan the call." not in fake.prompts[1]


def test_a_clear_thinking_edit_with_no_keep_leaves_the_templates_default(serve):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.gemma(), "Hi.")
    body = {**EARLIER_CALL, "context_management": {"edits": [{"type": "clear_thinking_20251015"}]}}

    assert http.post("/v1/messages", json=body).status_code == 200
    assert "Plan the call." not in fake.prompts[0]


def test_keeping_all_thinking_is_met_by_a_template_without_the_switch(serve):
    http, _ = serve(AnthropicAdapter(), mila.ChatProtocol.llama31(), "Hi.")

    assert http.post("/v1/messages", json={**EARLIER_CALL, **KEEP_ALL}).status_code == 200


@pytest.mark.parametrize("edit", [
    {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}},
    {"type": "clear_tool_uses_20250919"},
    {"type": "compact_20260112"},
])
def test_a_context_edit_that_would_clear_history_is_refused(serve, edit):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.gemma(), "Hi.")

    response = http.post("/v1/messages", json={**EARLIER_CALL, "context_management": {"edits": [edit]}})

    assert response.status_code == 400
    assert "context_management.edits[0]" in response.text
    assert fake.prompts == []


def test_omitted_thinking_travels_in_the_signature(serve):
    http, _ = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    body = {"max_tokens": 64, "thinking": {"type": "adaptive", "display": "omitted"},
            "messages": [{"role": "user", "content": "Weather?"}]}

    reply = http.post("/v1/messages", json=body).json()
    events = _events(http.post("/v1/messages", json={**body, "stream": True}).text)
    signatures = [event["delta"]["signature"] for event in events if event.get("delta", {}).get("type") == "signature_delta"]
    thinking = [event["delta"]["thinking"] for event in events if event.get("delta", {}).get("type") == "thinking_delta"]

    assert reply["content"][0] == {"type": "thinking", "thinking": "", "signature": "The tool says sunny."}
    assert signatures == ["The tool says sunny."] and thinking == [""]


def test_a_thinking_block_sent_back_with_only_a_signature_is_the_turns_reasoning(serve):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    body = {"max_tokens": 64, "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": "Need the tool."},
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"}]},
    ]}

    http.post("/v1/messages", json=body)

    assert "<think>\nNeed the tool.\n</think>" in fake.prompts[0]


# ---------------------------------------------------------------------------
# System text wherever an API places it (InferenceServerWire.md 3.6)
# ---------------------------------------------------------------------------

def test_claude_codes_system_message_after_the_user_joins_the_leading_turn(serve):
    # Claude Code 2.1.300's shape: the system field, the user's message, then a system message with the environment.
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    body = {"max_tokens": 64, "system": [{"type": "text", "text": "Be brief."}], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Codename?"}]},
        {"role": "system", "content": [{"type": "text", "text": "Working directory: /repo", "cache_control": {"type": "ephemeral"}}]},
    ]}

    response = http.post("/v1/messages", json=body)

    assert response.status_code == 200
    assert fake.prompts[0].count("<|im_start|>system") == 1
    assert "Be brief.\n\nWorking directory: /repo<|im_end|>\n<|im_start|>user\nCodename?" in fake.prompts[0]


def test_codexs_developer_messages_join_one_system_turn(serve):
    # Codex 0.162.1's shape: two developer messages, then the environment and the task as user messages.
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    body = {"input": [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are a coding agent."}]},
        {"type": "message", "role": "developer", "content": [
            {"type": "input_text", "text": "Skills."}, {"type": "input_text", "text": "Permissions."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context/>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Codename?"}]},
    ]}

    response = http.post("/v1/responses", json=body)

    assert response.status_code == 200
    assert fake.prompts[0].count("<|im_start|>system") == 1
    assert "You are a coding agent.\n\nSkills.Permissions.<|im_end|>" in fake.prompts[0]


LATER_SYSTEM = {"messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello."},
    {"role": "system", "content": "From now on, answer in French."},
    {"role": "user", "content": "Weather?"},
]}


def test_a_system_message_after_a_reply_stays_where_it_was_sent(serve):
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.gemma(), "Beau.")

    assert http.post("/v1/chat/completions", json=LATER_SYSTEM).status_code == 200
    assert "Hello.<turn|>\n<|turn>system\nFrom now on, answer in French.<turn|>" in fake.prompts[0]
    assert not fake.prompts[0].startswith("<bos><|turn>system")


def test_a_template_that_cannot_place_a_later_system_message_refuses_it(serve):
    http, fake = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)

    response = http.post("/v1/chat/completions", json=LATER_SYSTEM)

    assert response.status_code == 400
    assert "system turn must be the first" in response.text
    assert fake.prompts == []


def test_a_later_system_message_with_no_text_is_dropped(serve):
    http, _ = serve(OpenAIAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    body = {"messages": [dict(message, content="") if message["role"] == "system" else message for message in LATER_SYSTEM["messages"]]}

    assert http.post("/v1/chat/completions", json=body).status_code == 200


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": "First."}, {"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}, {"role": "system", "content": "Late."}],
    [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Between."}, {"role": "user", "content": "Again"}],
    [{"role": "user", "content": "Hi"}, {"role": "system", "content": "Turn.", "clear_at": "next_user_message"}],
    [{"role": "user", "content": "Hi"}, {"role": "system", "content": "", "output_config": {"effort": "low"}}],
])
def test_messages_refuses_a_system_message_anthropic_would_refuse(serve, messages):
    http, _ = serve(AnthropicAdapter(), mila.ChatProtocol.gemma(), "Hi.")

    response = http.post("/v1/messages", json={"max_tokens": 64, "messages": messages})

    assert response.status_code == 400
    assert "messages[" in response.text


def test_messages_places_consecutive_system_messages_together(serve):
    http, fake = serve(AnthropicAdapter(), mila.ChatProtocol.qwen(), QWEN_ANSWER)
    body = {"max_tokens": 64, "messages": [
        {"role": "user", "content": "Hi"}, {"role": "system", "content": "One."}, {"role": "system", "content": "Two."}]}

    assert http.post("/v1/messages", json=body).status_code == 200
    assert fake.prompts[0].count("<|im_start|>system") == 1
    assert "\n\nOne.\n\nTwo.<|im_end|>\n<|im_start|>user\nHi" in fake.prompts[0]


# ---------------------------------------------------------------------------
# W5: the server knows no family
# ---------------------------------------------------------------------------

def test_no_family_is_named_outside_the_loader():
    """Only model_worker.SESSION_FOR, the binding's loader table, may name a family (ModelHandle.md 3.5)."""
    source = pathlib.Path(__file__).resolve().parents[1] / "src" / "mila_llm_server"
    named = []

    for path in source.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\b(gemma|llama|qwen)\b", line, re.IGNORECASE) and path.name != "model_worker.py" \
                    and "Built with Llama" not in line and "gemma-4-12b-it-fp4" not in line:
                named.append(f"{path.name}:{number}: {line.strip()}")

    assert named == []
