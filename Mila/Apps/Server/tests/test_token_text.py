"""
Token ids become text as they stream: a character split across tokens comes out whole, and bytes that never form one
come out as U+FFFD rather than freezing the stream until the decode timeout ends the reply.
"""
from mila_llm_server.model_worker import TokenText

# A byte-level vocabulary: each id is one byte, as a byte-level BPE token can be.
BYTES = {index: bytes([index]) for index in range(256)}


def decode(ids, errors="strict"):
    return b"".join(BYTES[token] for token in ids).decode("utf-8", errors)


def streamed(ids):
    stream = TokenText(decode)
    pieces = [stream.feed(token) for token in ids]

    return [piece for piece in pieces if piece is not None], stream.invalid_runs


def test_a_character_split_across_tokens_comes_out_whole():
    assert streamed(list("é".encode()) + list(b"x")) == (["é", "x"], 0)


def test_bytes_that_never_form_a_character_do_not_freeze_the_stream():
    # A lone lead byte, then ordinary text: every strict decode of the buffer fails from here on.
    pieces, invalid = streamed([0xC3] + list(b"the rest of the reasoning"))

    assert "".join(pieces) == "�the rest of the reasoning"
    assert invalid == 1
