"""
The worker stops at a protocol's stop markers only outside the reasoning, where the parser reads calls. A marker the
model writes while thinking once ended Qwen 3.8's replies inside their reasoning, and the client got nothing. The
marker is the open of a result, never a call's close: a turn keeps every call the model makes before it.
"""
import mila

from mila_llm_server.model_worker import StopMarkerScan

QWEN = mila.ChatProtocol.qwen().markers
GEMMA = mila.ChatProtocol.gemma().markers


def scan(markers, begins_in_reasoning):
    return StopMarkerScan(markers["stop_after"], markers["reasoning_open"], markers["reasoning_close"],
                          begins_in_reasoning)


def first_stop(markers, chunks, begins_in_reasoning):
    scanner = scan(markers, begins_in_reasoning)

    for index, chunk in enumerate(chunks):
        if (matched := scanner.feed(chunk)) is not None:
            return index, matched

    return None


def test_closed_calls_do_not_stop_the_reply():
    qwen = ["<tool_call>", "<function=a>", "</function>", "</tool_call>", "\n", "<tool_call>", "<function=b>",
            "</function>", "</tool_call>"]
    gemma = ["<|tool_call>call:a{}<tool_call|>", "<|tool_call>call:b{}<tool_call|>", "<|tool_response>"]

    assert first_stop(QWEN, qwen, begins_in_reasoning=False) is None
    assert first_stop(GEMMA, gemma, begins_in_reasoning=False) == (2, "<|tool_response>")


def test_a_result_written_while_thinking_does_not_stop_the_reply():
    chunks = ["The ", "<tool_response>", " would say 3.", "</think>", "\n\n", "<tool_call>x</tool_call>",
              "<tool_response>"]

    assert first_stop(QWEN, chunks, begins_in_reasoning=True) == (6, "<tool_response>")


def test_a_result_mentioned_while_thinking_with_no_result_after_does_not_stop_the_reply():
    chunks = ["The <tool_response> said 3.", "</think>", "It is 3."]

    assert first_stop(QWEN, chunks, begins_in_reasoning=True) is None


def test_with_thinking_off_a_result_stops_at_once():
    assert first_stop(QWEN, ["Sure.", "<tool_call>x</tool_call>", "<tool_response>", "y"],
                      begins_in_reasoning=False) == (2, "<tool_response>")


def test_a_marker_split_across_chunks_is_found():
    assert first_stop(QWEN, ["</th", "ink>", "<tool_call>x</tool_call><tool_res", "ponse>"],
                      begins_in_reasoning=True) == (3, "<tool_response>")


def test_a_reopened_reasoning_span_is_skipped_too():
    chunks = ["</think>", "<think>", "<tool_response>", "</think>", "<tool_call>y</tool_call>", "<tool_response>"]

    assert first_stop(QWEN, chunks, begins_in_reasoning=True) == (5, "<tool_response>")


def test_gemma_channels_are_reasoning_wherever_they_open():
    chunks = ["Let me check.", "<|channel>", "thought\n<|tool_response>", "<channel|>",
              "<|tool_call>call:read{}<tool_call|>", "<|tool_response>"]

    assert first_stop(GEMMA, chunks, begins_in_reasoning=False) == (5, "<|tool_response>")
