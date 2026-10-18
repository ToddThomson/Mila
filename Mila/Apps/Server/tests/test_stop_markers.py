"""
The worker stops at a protocol's stop markers only outside the reasoning, where the parser reads calls. A call the
model drafts while thinking once ended Qwen 3.8's replies inside their reasoning, and the client got nothing.
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


def test_a_call_drafted_while_thinking_does_not_stop_the_reply():
    chunks = ["I could call ", "<tool_call>", "<function=read>", "</function>", "</tool_call>", " but first",
              "</think>", "\n\n", "<tool_call>", "<function=read>", "</function>", "</tool_call>"]

    assert first_stop(QWEN, chunks, begins_in_reasoning=True) == (11, "</tool_call>")


def test_a_result_mentioned_while_thinking_does_not_stop_the_reply():
    chunks = ["The <tool_response> said 3.", "</think>", "It is 3."]

    assert first_stop(QWEN, chunks, begins_in_reasoning=True) is None


def test_with_thinking_off_a_call_stops_at_once():
    assert first_stop(QWEN, ["Sure.", "<tool_call>", "x", "</tool_call>"], begins_in_reasoning=False) == (3, "</tool_call>")


def test_a_marker_split_across_chunks_is_found():
    assert first_stop(QWEN, ["</th", "ink>", "<tool_call>x</tool", "_call>"], begins_in_reasoning=True) == (3, "</tool_call>")


def test_a_reopened_reasoning_span_is_skipped_too():
    chunks = ["</think>", "<think>", "<tool_call>x</tool_call>", "</think>", "<tool_call>y</tool_call>"]

    assert first_stop(QWEN, chunks, begins_in_reasoning=True) == (4, "</tool_call>")


def test_gemma_channels_are_reasoning_wherever_they_open():
    chunks = ["Let me check.", "<|channel>", "thought\n<|tool_call>call:x{}<tool_call|>", "<channel|>",
              "<|tool_call>call:read{}<tool_call|>"]

    assert first_stop(GEMMA, chunks, begins_in_reasoning=False) == (4, "<tool_call|>")
