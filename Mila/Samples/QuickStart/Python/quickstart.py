"""
Mila Python quick start: one prompt in, generated tokens streamed out.

The smallest complete program that runs a local LLM with Mila -- load a model from the
store, encode a prompt, and stream the reply as it is produced. Single-shot by design:
no conversation history and no REPL, because neither teaches anything about Mila.
chat.py is where multi-turn and channel filtering live.

The C++ counterpart at ../Cpp/main.cpp does the same thing, with the same model, the
same template and the same defaults, so the two can be read side by side.

    python quickstart.py "Why is the sky blue?"
    python quickstart.py                          # prompts on stdin
"""

import sys

import common

# The published flagship. Loading never downloads, so an uninstalled name is an error
# rather than a surprise multi-gigabyte transfer -- pull it deliberately:
#   mila.ModelStore().pull(MODEL, mila.default_hub_owner())
MODEL = "gemma-4-12b-it-fp4"

# Well under the model's ceiling. Context length drives KV-cache VRAM, and a first run
# should fit comfortably rather than probe the limit.
CONTEXT_LENGTH = 4096

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
TOP_K = 40
TOP_P = 0.9

# Turns are <|turn>{role}\n{content}<turn|>, opened with <bos> and handed to the model
# with a bare <|turn>model\n. These are registered vocabulary tokens, so they are written
# as literal text and each encodes as one atomic token.
BOS = "<bos>"
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"

# Thinking off, and that takes two things. Omitting the <|think|> trigger deactivates it,
# but the 12B then emits "ghost" thought sections anyway -- priming an EMPTY thought
# channel suppresses them. Load-bearing, not cosmetic: without it the model narrates at
# you. With thinking ON you must not prime this, as it pre-empts real reasoning.
THOUGHT_PRIME = "<|channel>thought\n<channel|>"

# Gemma collapses to a single system instruction. A first run answers noticeably better
# with one than without.
SYSTEM_PROMPT = "You are a helpful assistant."


def build_prompt(user_message):
    """Wrap one user message in the Gemma 4 instruct template."""
    return (
        f"{BOS}"
        f"{TURN_OPEN}system\n{SYSTEM_PROMPT}{TURN_CLOSE}\n"
        f"{TURN_OPEN}user\n{user_message}{TURN_CLOSE}\n"
        f"{TURN_OPEN}model\n"
        f"{THOUGHT_PRIME}"
    )


def main():
    common.configure_console()

    user_message = " ".join(sys.argv[1:]).strip()

    if not user_message:
        try:
            user_message = input("Prompt: ").strip()
        except EOFError:
            user_message = ""

    if not user_message:
        print("No prompt given.", file=sys.stderr)
        return 1

    mila = common.import_mila("warning")

    print(f"Loading {MODEL} ...", flush=True)
    tokenizer, model, _record = common.load_from_store(mila, MODEL, CONTEXT_LENGTH)

    prompt_tokens = tokenizer.encode(build_prompt(user_message))

    # A token can carry only part of a multi-byte code point, so ids are held until they
    # decode cleanly. (The C++ sample needs no such buffer: its decode() yields bytes,
    # which concatenate into correct UTF-8 on their own. This is a Python str problem.)
    pending = []

    def on_token(token_id):
        pending.append(token_id)

        try:
            text = tokenizer.decode(pending)
        except UnicodeDecodeError:
            return

        pending.clear()
        sys.stdout.write(text)
        sys.stdout.flush()

    print()

    # The binding releases the GIL around generation, so this callback runs on a live
    # interpreter and the reply appears as it is produced rather than all at once.
    reason = model.generate(
        prompt_tokens,
        on_token,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        TOP_K,
        TOP_P,
    )

    # Why it stopped is the one outcome a caller cannot reconstruct from the tokens.
    print(f"\n\n[{reason}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
