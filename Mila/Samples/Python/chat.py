"""
Streaming chat with Gemma 4, in Python, with nothing hidden.

This is the whole loop: build the prompt in Gemma's instruct template, encode it,
stream tokens back, filter the reasoning channels out of the display, and hand the
turn to the model again. Everything Mila's C++ chat harness does around a bare
model, done here in one readable file.

    python chat.py
    python chat.py --temperature 0.6 --max-new-tokens 512

Ctrl-C stops generation without leaving the chat; /exit leaves.
"""

import argparse
import signal
import sys
import time

import common

# --- Gemma 4 instruct template (the Mila checkpoint) -----------------------
#
# Turns are <|turn>{role}\n{content}<turn|>, roles system / user / model, with a
# <bos> prefix and a <|turn>model\n primer to hand generation to the model. These
# are registered vocabulary tokens, so they are written as literal text and the
# tokenizer encodes each as one atomic token. NOT the Gemma 3 style
# <start_of_turn>/<end_of_turn>, which this vocabulary does not contain.
BOS = "<bos>"
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"
CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"

# Thinking off. Priming an EMPTY thought channel onto the prompt suppresses the
# ghost reasoning sections the 12B otherwise emits when thinking is deactivated.
# Load-bearing, not cosmetic -- without it the model narrates at you.
THOUGHT_PRIME = f"{CHANNEL_OPEN}thought\n{CHANNEL_CLOSE}"

# Registered tokens that must never reach the screen if one slips through.
CONTROL_TOKENS = (
    BOS, "<eos>", "<pad>", TURN_OPEN, TURN_CLOSE, "<|think|>",
    "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
    "<|tool_response>", "<tool_response|>", '<|"|>',
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def build_prompt(system_prompt, history, user_message):
    """
    Render the conversation into the Gemma instruct template.

    The full history is replayed every turn. That is not wasteful here: Mila
    reuses the KV-cache prefix that matches token-for-token, so only the new
    tokens actually prefill.
    """
    parts = [BOS]

    if system_prompt:
        parts.append(f"{TURN_OPEN}system\n{system_prompt}{TURN_CLOSE}\n")

    for role, content in history:
        parts.append(f"{TURN_OPEN}{role}\n{content}{TURN_CLOSE}\n")

    parts.append(f"{TURN_OPEN}user\n{user_message}{TURN_CLOSE}\n")
    parts.append(f"{TURN_OPEN}model\n")
    parts.append(THOUGHT_PRIME)

    return "".join(parts)


def strip_control_tokens(text):
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")

    return text


class AnswerStream:
    """
    Reduces the raw token stream to the answer text, as it arrives.

    Gemma 4 output is channel-structured: reasoning sits inside
    <|channel>label\\n...<channel|> and the answer is what surrounds it. Even with
    the empty thought prime the 12B opens a channel mid-answer sometimes, so every
    channel span is dropped wherever it appears rather than just a leading one.

    Text is held back from a trailing '<' until it is known not to be the start of
    a marker, since a decoded chunk can end part-way through one.
    """

    def __init__(self):
        self._buffer = ""
        self._in_channel = False

    def feed(self, text):
        """Return the displayable part of the stream so far."""
        self._buffer += text
        out = []

        while True:
            if self._in_channel:
                end = self._buffer.find(CHANNEL_CLOSE)

                if end == -1:
                    break

                self._buffer = self._buffer[end + len(CHANNEL_CLOSE):]
                self._in_channel = False
                continue

            start = self._buffer.find(CHANNEL_OPEN)

            if start == -1:
                emit, self._buffer = self._split_at_partial_marker(self._buffer)
                out.append(strip_control_tokens(emit))
                break

            out.append(strip_control_tokens(self._buffer[:start]))
            self._buffer = self._buffer[start + len(CHANNEL_OPEN):]
            self._in_channel = True

        return "".join(out)

    def flush(self):
        """Emit whatever was held back. Call once generation has finished."""
        tail = "" if self._in_channel else strip_control_tokens(self._buffer)
        self._buffer = ""

        return tail

    @staticmethod
    def _split_at_partial_marker(text):
        """Split into (safe to display, hold back) at a trailing partial marker."""
        cut = text.rfind("<")

        # A '<' with no '>' after it may still be growing into a marker. The length
        # bound keeps ordinary prose containing '<' from stalling the display.
        if cut != -1 and ">" not in text[cut:] and len(text) - cut < 16:
            return text[:cut], text[cut:]

        return text, ""


def stream_turn(mila, model, tokenizer, prompt, args):
    """
    Stream one model turn to stdout. Returns the answer text.

    Two things make this work from Python: the binding releases the GIL around
    generation, so the per-token callback runs on a live interpreter; and
    StopController is a cooperative cancel the decode loop checks each step.
    """
    prompt_tokens = tokenizer.encode(prompt)
    answer = AnswerStream()
    collected = []
    pending_ids = []
    counters = {"tokens": 0, "first_token_at": None}
    started = time.perf_counter()

    def on_token(token_id):
        counters["tokens"] += 1

        if counters["first_token_at"] is None:
            counters["first_token_at"] = time.perf_counter()

        # A token can carry only part of a multi-byte code point, so ids are held
        # until they decode cleanly.
        pending_ids.append(token_id)

        try:
            text = tokenizer.decode(pending_ids)
        except UnicodeDecodeError:
            return

        pending_ids.clear()
        visible = answer.feed(text)

        if visible:
            collected.append(visible)
            sys.stdout.write(visible)
            sys.stdout.flush()

    stop = mila.StopController()

    # Ctrl-C asks the decode loop to stop rather than raising through C++ with a
    # forward pass in flight. The handler is only reached while tokens are arriving
    # (that is when the callback hands the interpreter back a live GIL), so an
    # interrupt during prefill takes effect at the first token.
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop.request_stop())

    try:
        model.generate_streaming(
            prompt_tokens,
            on_token,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            stop,
        )
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    tail = answer.flush()

    if tail:
        collected.append(tail)
        sys.stdout.write(tail)

    sys.stdout.write("\n")

    if stop.stop_requested:
        sys.stdout.write("[stopped]\n")

    elapsed = time.perf_counter() - started
    generated = counters["tokens"]

    if args.stats and generated:
        prefill = counters["first_token_at"] - started
        decode_rate = (generated - 1) / max(elapsed - prefill, 1e-9)
        sys.stdout.write(
            f"[{len(prompt_tokens)} prompt tokens, {generated} generated, "
            f"{prefill:.2f}s to first token, {decode_rate:.1f} tok/s]\n"
        )

    sys.stdout.flush()

    return "".join(collected).strip()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--weights", help="Path to the Mila .bin weights (default: Data/Models/Gemma).")
    parser.add_argument("--tokenizer", help="Path to the Gemma tokenizer .bin.")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="KV-cache depth to build for. Larger costs VRAM (default: 4096).")
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device ordinal.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System instruction.")
    parser.add_argument("--log-level", default="warning", choices=("trace", "info", "warning", "error"))
    parser.add_argument("--stats", action="store_true", help="Print timing after each turn.")

    return parser.parse_args()


def main():
    args = parse_args()

    common.configure_console()
    mila = common.import_mila(args.log_level)
    weights, tokenizer_path = common.resolve_paths("gemma", args.weights, args.tokenizer)

    print(f"Loading {weights.name} (FP4, context {args.context_length}) ...", flush=True)
    load_started = time.perf_counter()
    tokenizer, model = common.load(
        mila, "gemma", weights, tokenizer_path, args.context_length, args.device_index)
    print(f"Ready in {time.perf_counter() - load_started:.1f}s. "
          f"Ctrl-C stops a response, /exit quits, /clear forgets the conversation.\n")

    history = []

    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue

        if user_message in ("/exit", "/quit"):
            break

        if user_message == "/clear":
            history.clear()
            print("[conversation cleared]\n")
            continue

        print("\ngemma> ", end="", flush=True)
        prompt = build_prompt(args.system, history, user_message)
        answer = stream_turn(mila, model, tokenizer, prompt, args)
        print()

        history.append(("user", user_message))
        history.append(("model", answer))


if __name__ == "__main__":
    main()
