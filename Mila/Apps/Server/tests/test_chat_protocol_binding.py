"""
The binding's ChatProtocol object, the one protocol the server will hold for its loaded model
(InferenceServerWire.md Phase 2).

The prompts themselves are held to each checkpoint's template by the library's own suite
(Mila/Tests/Dnn/Models/*/*.ChatProtocol.cpp); these check that the projection carries every field
across, in both directions, and refuses what the library refuses.
"""
import json

import mila
import pytest

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
                                            "parameters": {"type": "object",
                                                           "properties": {"city": {"type": "string"}}}}}

RESUME = [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Paris"}'}]},
    {"role": "tool", "content": "Sunny", "tool_name": "get_weather"},
]


def test_every_template_has_a_protocol():
    for protocol in (mila.ChatProtocol.gemma(), mila.ChatProtocol.qwen(),
                     mila.ChatProtocol.llama31(), mila.ChatProtocol.llama32()):
        prompt = protocol.format_prompt([{"role": "user", "content": "hi"}])
        assert prompt["text"]


def test_a_tool_result_renders_under_its_own_name():
    prompt = mila.ChatProtocol.gemma().format_prompt(RESUME, tools_json=json.dumps([WEATHER]))

    assert prompt["text"].endswith(
        '<|tool_response>response:get_weather{value:<|"|>Sunny<|"|>}<tool_response|>')


def test_template_variables_are_the_templates_own():
    variables = {v["name"]: v for v in mila.ChatProtocol.qwen().template_variables}

    assert variables["reasoning_effort"]["default"] == "xhigh"
    assert variables["reasoning_effort"]["values"] == ["xhigh", "medium", "low"]
    assert mila.ChatProtocol.qwen().format_prompt(
        [{"role": "user", "content": "hi"}])["reply_begins_in_reasoning"] is True
    assert mila.ChatProtocol.qwen().format_prompt(
        [{"role": "user", "content": "hi"}],
        template_variables={"enable_thinking": "false"})["reply_begins_in_reasoning"] is False


def test_a_value_the_template_does_not_accept_is_refused():
    with pytest.raises(ValueError, match="xhigh"):
        mila.ChatProtocol.qwen().format_prompt(
            [{"role": "user", "content": "hi"}], template_variables={"reasoning_effort": "high"})


def test_a_reply_carries_every_call_and_reports_a_malformed_one():
    protocol = mila.ChatProtocol.gemma()
    prompt = protocol.format_prompt([{"role": "user", "content": "hi"}])
    reply = protocol.parse_reply(
        '<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:2}<tool_call|>'
        '<|tool_call>call:c{cmd="ls"}<tool_call|>', prompt)

    assert [call["name"] for call in reply["tool_calls"]] == ["a", "b"]
    assert json.loads(reply["tool_calls"][1]["arguments"]) == {"y": 2}
    assert reply["malformed_tool_calls"] == ['<|tool_call>call:c{cmd="ls"}<tool_call|>']


def test_a_reply_read_against_its_prompt_splits_its_reasoning():
    protocol = mila.ChatProtocol.qwen()
    prompt = protocol.format_prompt([{"role": "user", "content": "hi"}])
    reply = protocol.parse_reply("Thinking it over.\n</think>\n\nHello.", prompt)

    assert reply["reasoning"] == "Thinking it over."
    assert reply["content"] == "Hello."


def test_markers_carry_the_stops():
    markers = mila.ChatProtocol.qwen().markers

    assert markers["stop_after"] == ["</tool_call>", "<tool_response>"]
    assert markers["reasoning_close"] == "</think>"


@pytest.mark.parametrize("factory", ["gemma", "qwen", "llama31", "llama32"])
def test_each_template_says_whether_it_renders_a_later_system_turn(factory):
    protocol = getattr(mila.ChatProtocol, factory)()
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
               {"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Weather?"}]

    if protocol.renders_later_system_turns:
        assert "Answer in French." in protocol.format_prompt(history)["text"]
    else:
        with pytest.raises(ValueError, match="system"):
            protocol.format_prompt(history)

    assert protocol.renders_later_system_turns == (factory != "qwen")


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="role"):
        mila.ChatProtocol.gemma().format_prompt([{"role": "developer", "content": "hi"}])
