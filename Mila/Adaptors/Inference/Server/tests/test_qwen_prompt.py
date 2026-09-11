"""
MIS's Qwen 3.8 path renders through the runtime, not through a template of its own.

`prompt.py` calls `mila.qwen_format_prompt`, which is the projection of
Dnn.Models.QwenProtocol -- the same module the chat harness renders through. So these are
not parity tests against a second implementation; there is no second implementation. They pin
that MIS hands the runtime the right conversation and does not reshape what comes back.

The golden strings are the C++ renderer's output. They are duplicated in
Mila/Tests/Dnn/Models/Qwen/Qwen.Protocol.cpp, which tests the same function
directly -- if these two ever disagree, the projection is what broke.
"""
import dataclasses

import mila
import pytest

from mila_llm_server import prompt
from mila_llm_server.config import ModelFamily, loaded


@pytest.fixture(autouse=True)
def qwen_family():
    """config.loaded is a module-level singleton the whole server reads, so a family a test
    sets has to be put back or every later test sees it."""
    before = dataclasses.replace(loaded)
    loaded.family = ModelFamily.qwen

    yield

    for field in dataclasses.fields(before):
        setattr(loaded, field.name, getattr(before, field.name))


EXPECTED_MINIMAL = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

EXPECTED_WITH_HISTORY = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n"
    "<|im_start|>user\nAnd 3 + 3?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def test_minimal_turn():
    assert prompt.build_instruct_prompt("What is 2 + 2?") == EXPECTED_MINIMAL


def test_history_assistant_turn_carries_an_empty_reasoning_span():
    rendered = prompt.build_instruct_prompt(
        "And 3 + 3?",
        history=[
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
    )

    assert rendered == EXPECTED_WITH_HISTORY


def test_no_bos_marker():
    # The checkpoint sets add_bos_token false. Gemma's template opens with <bos> and
    # Llama's with <|begin_of_text|>, so the failure mode is a template copied from either.
    rendered = prompt.build_instruct_prompt("hello")

    assert rendered.startswith("<|im_start|>")
    assert "<bos>" not in rendered
    assert "<|begin_of_text|>" not in rendered


def test_history_system_turn_is_dropped_in_favour_of_the_assembled_one():
    # The runtime assembles the system turn in the order the template puts its parts in; a
    # system message replayed out of history would produce a second one.
    rendered = prompt.build_instruct_prompt(
        "hello",
        system_prompt="Be terse.",
        history=[{"role": "system", "content": "Be verbose."}],
    )

    assert rendered.count("<|im_start|>system") == 1
    assert "Be verbose." not in rendered


def test_tools_render_in_the_trained_section():
    # The <tools> section carries the call-format specification the model emits <tool_call>
    # spans against. Declaring tools as plain JSON instead is what makes a model improvise an
    # off-spec call, so the trained form is the whole point of going through the runtime.
    rendered = prompt.build_instruct_prompt(
        "weather?",
        tools=[{"name": "get_weather", "description": "Look up weather"}],
    )

    assert "<tools>" in rendered
    assert '{"description":"Look up weather","name":"get_weather"}' in rendered
    assert "<tool_call>\n<function=example_function_name>" in rendered


def test_no_tools_means_no_tools_section():
    # The absence of the section is what tells the model there are none.
    assert "<tools>" not in prompt.build_instruct_prompt("hello")


def test_the_runtime_is_the_renderer():
    # The property this whole file exists to hold: MIS adds nothing between the conversation
    # and the template. A prompt built here equals the runtime's for the same turns.
    turns = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    assert prompt.build_instruct_prompt("What is 2 + 2?") == mila.qwen_format_prompt(
        turns, enable_thinking=False
    )


def test_tool_results_replay_as_a_user_turn_carrying_a_response_span():
    # Not a role of its own -- rendering "tool" by name would open a role the model was never
    # trained to read. MIS passes the role through and the runtime decides the rendering.
    rendered = mila.qwen_format_prompt(
        [
            {"role": "user", "content": "weather?"},
            {"role": "tool", "content": "sunny"},
        ],
        enable_thinking=False,
    )

    assert "<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" in rendered


def test_an_unknown_role_is_refused_rather_than_guessed():
    with pytest.raises(RuntimeError, match="not a conversation role"):
        mila.qwen_format_prompt([{"role": "developer", "content": "hi"}])
