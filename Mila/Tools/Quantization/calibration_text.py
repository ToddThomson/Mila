"""The calibration text for an instruct model's fit: tool-call conversations, chat and prose, each conversation
rendered through the model's own chat template so the fit sees the prompt format the model is served in.

Qwen3.8.md ("The test is a re-fit on calibration text of the kind the model is served") records the sources
and why. The parts are mixed by token share and interleaved document by document, so any window
pack_qwen.py draws holds a mix rather than a run of one kind.

Two things are kept out, and the output is refused if either survives: any document naming a tool of
tool_name_probe.py's probe sets, and any conversation that asks a question BFCL asks (ModelEval.md Phase 8
scores the build on BFCL). Run in the Evaluation venv, which has the parquet reader and bfcl-eval's data.

    python calibration_text.py --hermes corpus/hermes-function-calling-v1/func-calling.json \\
        --ultrachat corpus/ultrachat_200k/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet \\
        --prose corpus/wiki.train.raw --output corpus/calibration_qwen38.txt
"""

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

from tool_name_probe import probe_names

FUNCTION_CALLING_SYSTEM = "You are a function calling AI model."
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
PROSE_HEADING = re.compile(r"^ = [^=].* = $", re.MULTILINE)


class Unusable(Exception):
    """A source record this tool cannot turn into a conversation the template renders."""


def name_pattern(names):
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))

    return re.compile(rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])")


def normalized(text):
    return " ".join(text.split()).casefold()


def bfcl_questions(data_dir):
    """Every user turn BFCL asks, normalized."""
    questions = set()

    for path in sorted(Path(data_dir).glob("BFCL_*.json")):
        text = path.read_text(encoding="utf-8")

        # Most files are one entry a line; a few are one JSON document.
        try:
            whole = json.loads(text)
            entries = whole if isinstance(whole, list) else [whole]
        except json.JSONDecodeError:
            entries = [json.loads(line) for line in text.splitlines() if line.strip()]

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            for turn in entry.get("question", []):
                for message in turn:
                    if message.get("role") == "user" and isinstance(message.get("content"), str):
                        questions.add(normalized(message["content"]))

    if not questions:
        raise SystemExit(f"no BFCL questions found under {data_dir}")

    return questions


def default_bfcl_data():
    spec = importlib.util.find_spec("bfcl_eval")

    if spec is None:
        raise SystemExit("bfcl-eval is not importable here; pass --bfcl-data with its data directory")

    return Path(spec.origin).parent / "data"


def tool_call(block):
    try:
        call = json.loads(block)
    except json.JSONDecodeError as error:
        raise Unusable(f"call is not JSON: {error}") from error

    arguments = call.get("arguments")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise Unusable(f"arguments are not JSON: {error}") from error

    if not isinstance(call.get("name"), str) or not isinstance(arguments, dict):
        raise Unusable("call has no name or no argument object")

    return {"type": "function", "function": {"name": call["name"], "arguments": arguments}}


def hermes_conversation(record):
    """(messages, tools) for one Hermes record, its own system prompt replaced by the template's tools."""
    turns = record["conversations"]

    if not turns or turns[0]["from"] != "system" or not turns[0]["value"].startswith(FUNCTION_CALLING_SYSTEM):
        raise Unusable("not a function-calling conversation")

    tools = json.loads(record["tools"]) if isinstance(record["tools"], str) else record["tools"]
    messages = []

    for turn in turns[1:]:
        text = turn["value"]

        if turn["from"] == "human":
            messages.append({"role": "user", "content": text})
        elif turn["from"] == "gpt":
            calls = [tool_call(block) for block in TOOL_CALL.findall(text)]
            message = {"role": "assistant", "content": TOOL_CALL.sub("", text).strip()}

            if calls:
                message["tool_calls"] = calls

            messages.append(message)
        elif turn["from"] == "tool":
            responses = TOOL_RESPONSE.findall(text)

            if not responses:
                raise Unusable("tool turn has no response")

            for block in responses:
                try:
                    response = json.loads(block)
                except json.JSONDecodeError as error:
                    raise Unusable(f"response is not JSON: {error}") from error

                content = response.get("content", response)
                messages.append({"role": "tool", "content": content if isinstance(content, str) else json.dumps(content)})
        else:
            raise Unusable(f"unknown role {turn['from']!r}")

    return messages, tools


def prose_documents(text):
    """Wikitext articles, split at their top-level headings."""
    starts = [match.start() for match in PROSE_HEADING.finditer(text)] + [len(text)]

    return [text[begin:end].strip() + "\n" for begin, end in zip(starts, starts[1:]) if text[begin:end].strip()]


def asks_bfcl(messages, questions):
    return any(message["role"] == "user" and normalized(message["content"]) in questions for message in messages)


def take(documents, budget, tokenizer, excluded, names, kind):
    """
    Documents in order until the kind's token budget is reached, skipping any that names a probed tool.
    Returns the chosen documents, their tokens, and the documents after the last one looked at.
    """
    chosen, tokens, looked_at = [], 0, 0

    for text in documents:
        if tokens >= budget:
            break

        looked_at += 1

        if names.search(text):
            excluded[f"{kind}: names a probed tool"] = excluded.get(f"{kind}: names a probed tool", 0) + 1
            continue

        chosen.append(text)
        tokens += len(tokenizer(text, add_special_tokens=False).input_ids)

    if tokens < budget:
        raise SystemExit(f"{kind}: sources hold {tokens} usable tokens, short of the {budget} asked for")

    return chosen, tokens, documents[looked_at:]


HELD_OUT_SEPARATOR = "\x1e"


def write_held_out(path, remaining, budget, tokenizer, names, calibration, kind):
    """
    Documents of one kind that the calibration text never saw, for scoring a fit on served text. Each is a
    conversation from its own start, separated by HELD_OUT_SEPARATOR, so a scorer reads every one cold: laid
    end to end, Qwen 3.8 FP4 scores the same conversations worse the more of them precede (Untriaged.md).
    """
    excluded = {}
    chosen, tokens, _ = take(remaining, budget, tokenizer, excluded, names, kind)
    overlap = sum(document in calibration for document in chosen)

    if overlap:
        raise SystemExit(f"refused: {overlap} held-out {kind} documents are also in the calibration text")

    if any(HELD_OUT_SEPARATOR in document for document in chosen):
        raise SystemExit(f"refused: a held-out {kind} document holds the separator itself")

    Path(path).write_text(HELD_OUT_SEPARATOR.join(chosen), encoding="utf-8", newline="\n")
    print(f"held out {kind:5s} {len(chosen):6d} documents {tokens:9d} tokens -> {path}")


def build(args):
    from transformers import AutoTokenizer
    import pyarrow.parquet

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    names = name_pattern(probe_names())
    questions = bfcl_questions(args.bfcl_data or default_bfcl_data())
    generator = random.Random(args.seed)
    excluded = {}

    def render(messages, tools=None):
        return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)

    tool_documents = []

    for record in json.loads(Path(args.hermes).read_text(encoding="utf-8")):
        try:
            messages, tools = hermes_conversation(record)
        except Unusable as reason:
            key = f"tool: {str(reason).split(':')[0]}"
            excluded[key] = excluded.get(key, 0) + 1
            continue

        if asks_bfcl(messages, questions):
            excluded["tool: asks a BFCL question"] = excluded.get("tool: asks a BFCL question", 0) + 1
            continue

        tool_documents.append(render(messages, tools))

    chat_documents = []

    for row in pyarrow.parquet.read_table(args.ultrachat, columns=["messages"]).to_pylist():
        messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]

        if asks_bfcl(messages, questions):
            excluded["chat: asks a BFCL question"] = excluded.get("chat: asks a BFCL question", 0) + 1
            continue

        chat_documents.append(render(messages))

    prose = prose_documents(Path(args.prose).read_text(encoding="utf-8"))

    for documents in (tool_documents, chat_documents, prose):
        generator.shuffle(documents)

    tool_budget = int(args.tokens * args.tool_share)
    chat_budget = int(args.tokens * args.chat_share)
    prose_budget = args.tokens - tool_budget - chat_budget

    if min(tool_budget, chat_budget, prose_budget) < 0:
        raise SystemExit("--tool-share and --chat-share must sum to at most 1")

    tool_chosen, tool_tokens, tool_remaining = take(tool_documents, tool_budget, tokenizer, excluded, names, "tool")
    chat_chosen, chat_tokens, chat_remaining = take(chat_documents, chat_budget, tokenizer, excluded, names, "chat")
    prose_chosen, prose_tokens, _ = take(prose, prose_budget, tokenizer, excluded, names, "prose")

    mixed = tool_chosen + chat_chosen + prose_chosen
    generator.shuffle(mixed)
    text = "".join(document if document.endswith("\n") else document + "\n" for document in mixed)

    survivor = names.search(text)

    if survivor:
        raise SystemExit(f"refused: the output names the probed tool {survivor.group(0)!r}")

    output = Path(args.output)
    # As rendered: a reader that takes bytes, as the C++ measurement harness does, must see the template's
    # own line endings, and a text-mode write on Windows would make every one of them CRLF.
    output.write_text(text, encoding="utf-8", newline="\n")
    total = len(tokenizer(text, add_special_tokens=False).input_ids)

    print(f"tool  {len(tool_chosen):6d} documents {tool_tokens:9d} tokens")
    print(f"chat  {len(chat_chosen):6d} documents {chat_tokens:9d} tokens")
    print(f"prose {len(prose_chosen):6d} documents {prose_tokens:9d} tokens")
    print(f"total {len(mixed):6d} documents {total:9d} tokens, joined")

    for reason, count in sorted(excluded.items()):
        print(f"excluded {count:6d}  {reason}")

    print(f"{output}  sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}")

    calibration = set(mixed)

    if args.held_out_tool:
        write_held_out(args.held_out_tool, tool_remaining, args.held_out_tokens, tokenizer, names, calibration, "tool")

    if args.held_out_chat:
        write_held_out(args.held_out_chat, chat_remaining, args.held_out_tokens, tokenizer, names, calibration, "chat")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B", help="checkpoint whose tokenizer and chat template render the text")
    parser.add_argument("--hermes", required=True, help="hermes-function-calling-v1 func-calling.json")
    parser.add_argument("--ultrachat", required=True, help="an ultrachat_200k parquet shard")
    parser.add_argument("--prose", required=True, help="wikitext-2-raw wiki.train.raw")
    parser.add_argument("--bfcl-data", help="bfcl-eval's data directory (default: the installed package's)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=600_000, help="total tokens before joining (default: 600000)")
    parser.add_argument("--tool-share", type=float, default=0.5, help="share of tool-call conversations (default: 0.5)")
    parser.add_argument("--chat-share", type=float, default=0.3, help="share of chat; prose takes the rest (default: 0.3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-tool", help="also write tool-call conversations the calibration text does not hold")
    parser.add_argument("--held-out-chat", help="also write chat the calibration text does not hold")
    parser.add_argument("--held-out-tokens", type=int, default=48_000,
                        help="tokens in each held-out file (default: 48000)")
    build(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
