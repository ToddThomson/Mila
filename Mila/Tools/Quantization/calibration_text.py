"""The calibration text for an instruct model's fit: tool-call conversations, chat, code, multilingual chat and
prose, each conversation rendered through the model's own chat template so the fit sees the prompt format the
model is served in.

Qwen3.8.md ("The test is a re-fit on calibration text of the kind the model is served") records the sources
and why. The parts are mixed by token share and interleaved document by document, so any window
pack_qwen.py draws holds a mix rather than a run of one kind.

Code is rendered where an agent meets it: as the content of a write_file call, as a file an open_file call
returns, as a fenced block in a chat answer, and a shell script also as a terminal call. Files are split
between calibration and held-out before they are cut into pieces, so no held-out piece shares a file with
the calibration text.

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


# open_file, not read_file: read_file is one of tool_name_probe.py's held-out names.
CODE_TOOLS = [
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file, replacing it.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "open_file", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "terminal", "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

# Suffix -> the fence language a chat answer names. Anything else in the repository is not code for this text.
CODE_LANGUAGES = {
    ".py": "python", ".sh": "bash", ".ps1": "powershell", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".cmake": "cmake", ".ixx": "cpp", ".cpp": "cpp", ".h": "cpp", ".cu": "cuda", ".cuh": "cuda",
}
SHELL_LANGUAGES = {"bash", "powershell"}

# Repository paths that are not Mila's own source, or are what the build is scored on.
REPOSITORY_EXCLUDED = ("Data/", "Dev/", "out/", ".git/", "Mila/Tools/Evaluation/", "Mila/Tools/Quantization/corpus/",
                       "node_modules/", ".venv/", "__pycache__/")
STDLIB_EXCLUDED = ("test/", "idlelib/", "site-packages/", "__pycache__/", "lib2to3/", "tkinter/test/")
MAXIMUM_FILE_BYTES = 100_000
PIECE_LINES = 120


def code_files(root, suffixes, excluded):
    """(relative path, language, text) for every source file under root, in path order."""
    root = Path(root)
    files = []

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        language = "cmake" if path.name == "CMakeLists.txt" else CODE_LANGUAGES.get(path.suffix)

        if (not path.is_file() or language is None or language not in suffixes
                or any(relative.startswith(prefix) or f"/{prefix}" in f"/{relative}" for prefix in excluded)
                or path.stat().st_size > MAXIMUM_FILE_BYTES):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if text.strip():
            files.append((relative, language, text))

    return files


def code_pieces(text):
    """A file cut into pieces of at most PIECE_LINES lines, each cut at a line that starts at column zero."""
    pieces, current = [], []

    for line in text.splitlines():
        at_top_level = bool(line) and not line[0].isspace()

        if current and (len(current) >= PIECE_LINES or (at_top_level and len(current) >= PIECE_LINES // 2)):
            pieces.append("\n".join(current).strip("\n") + "\n")
            current = []

        current.append(line)

    if "".join(current).strip():
        pieces.append("\n".join(current).strip("\n") + "\n")

    return pieces


def code_conversation(path, language, piece, generator):
    """One piece of code, as an agent meets it."""
    shapes = ["write", "read", "chat"] + (["terminal"] if language in SHELL_LANGUAGES and piece.count("\n") <= 30 else [])
    shape = generator.choice(shapes)

    if shape == "write":
        call = {"type": "function", "function": {"name": "write_file", "arguments": {"path": path, "content": piece}}}

        return [{"role": "user", "content": f"Write {path}."},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": json.dumps({"bytes_written": len(piece.encode("utf-8"))})},
                {"role": "assistant", "content": f"Wrote {path}."}]

    if shape == "read":
        call = {"type": "function", "function": {"name": "open_file", "arguments": {"path": path}}}

        return [{"role": "user", "content": f"Open {path}."},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": piece},
                {"role": "assistant", "content": f"That is {path}."}]

    if shape == "terminal":
        call = {"type": "function", "function": {"name": "terminal", "arguments": {"command": piece.rstrip("\n")}}}

        return [{"role": "user", "content": "Run it."},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": json.dumps({"output": "", "exit_code": 0})},
                {"role": "assistant", "content": "It ran."}]

    return [{"role": "user", "content": f"Show me {path}."},
            {"role": "assistant", "content": f"```{language}\n{piece}```"}]


def multilingual_conversations(parquet_path):
    """oasst1's best-ranked thread from each root, for every root not in English, as (language, messages)."""
    import pyarrow.parquet

    rows = pyarrow.parquet.read_table(
        parquet_path, columns=["message_id", "parent_id", "role", "text", "lang", "rank"]).to_pylist()
    children = {}

    for row in rows:
        children.setdefault(row["parent_id"], []).append(row)

    conversations = []

    for root in children.get(None, []):
        if root["lang"] == "en":
            continue

        messages, node = [], root

        while node is not None:
            messages.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
            replies = sorted(children.get(node["message_id"], []),
                             key=lambda reply: reply["rank"] if reply["rank"] is not None else 1_000)
            node = replies[0] if replies else None

        # A thread the template renders ends on the assistant.
        if messages[-1]["role"] == "user":
            messages.pop()

        if len(messages) >= 2:
            conversations.append(messages)

    return conversations


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

    # Code: files split between calibration and held-out first, so no held-out piece shares a file with the text.
    # Its own generator, so adding code leaves the tool and chat documents chosen in the order they were: a smaller
    # share of them is then a subset of an earlier build's, and that build's held-out files stay unseen.
    code_generator = random.Random(args.seed + 1)
    held_out_code_documents = []
    sources = []

    if args.python_source:
        sources.append(("python", code_files(args.python_source, {"python"}, STDLIB_EXCLUDED), args.python_share))

    if args.repository_source:
        repository_languages = set(CODE_LANGUAGES.values()) - {"python"}
        sources.append(("repository", code_files(args.repository_source, repository_languages, REPOSITORY_EXCLUDED),
                        1.0 - args.python_share))

    code_parts = []

    for source, files, share in sources:
        code_generator.shuffle(files)
        held_out_count = max(1, len(files) // 10)
        render_files = lambda chosen: [render(code_conversation(f"{source}/{path}", language, piece, code_generator),
                                              CODE_TOOLS)
                                       for path, language, text in chosen for piece in code_pieces(text)]
        calibration_pieces = render_files(files[held_out_count:])
        code_generator.shuffle(calibration_pieces)
        code_parts.append((source, calibration_pieces, share))
        held_out_code_documents += render_files(files[:held_out_count])
        print(f"code  {source:10s} {len(files):6d} files, {held_out_count} held out")

    multilingual_documents = []

    if args.multilingual:
        for messages in multilingual_conversations(args.multilingual):
            if asks_bfcl(messages, questions):
                excluded["multilingual: asks a BFCL question"] = excluded.get("multilingual: asks a BFCL question", 0) + 1
                continue

            multilingual_documents.append(render(messages))

    for documents in (tool_documents, chat_documents, prose, multilingual_documents, held_out_code_documents):
        generator.shuffle(documents)

    tool_budget = int(args.tokens * args.tool_share)
    chat_budget = int(args.tokens * args.chat_share)
    code_budget = int(args.tokens * args.code_share)
    multilingual_budget = int(args.tokens * args.multilingual_share)
    prose_budget = args.tokens - tool_budget - chat_budget - code_budget - multilingual_budget

    if min(tool_budget, chat_budget, code_budget, multilingual_budget, prose_budget) < 0:
        raise SystemExit("the shares must sum to at most 1")

    if code_budget and not code_parts:
        raise SystemExit("--code-share needs --python-source or --repository-source")

    if multilingual_budget and not multilingual_documents:
        raise SystemExit("--multilingual-share needs --multilingual")

    tool_chosen, tool_tokens, tool_remaining = take(tool_documents, tool_budget, tokenizer, excluded, names, "tool")
    chat_chosen, chat_tokens, chat_remaining = take(chat_documents, chat_budget, tokenizer, excluded, names, "chat")
    code_chosen, code_tokens = [], 0
    total_share = sum(share for _, _, share in code_parts)

    for source, pieces, share in code_parts:
        chosen, tokens, _ = take(pieces, int(code_budget * share / total_share), tokenizer, excluded, names,
                                 f"code ({source})")
        code_chosen += chosen
        code_tokens += tokens

    multilingual_chosen, multilingual_tokens, _ = take(multilingual_documents, multilingual_budget, tokenizer, excluded,
                                                       names, "multilingual")
    prose_chosen, prose_tokens, _ = take(prose, prose_budget, tokenizer, excluded, names, "prose")

    mixed = tool_chosen + chat_chosen + code_chosen + multilingual_chosen + prose_chosen
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
    print(f"code  {len(code_chosen):6d} documents {code_tokens:9d} tokens")
    print(f"multi {len(multilingual_chosen):6d} documents {multilingual_tokens:9d} tokens")
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

    if args.held_out_code:
        write_held_out(args.held_out_code, held_out_code_documents, args.held_out_tokens, tokenizer, names, calibration,
                       "code")


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
    parser.add_argument("--python-source", help="a Python standard library directory (its Lib), for code")
    parser.add_argument("--repository-source", help="a Mila checkout, for its shell, configuration and C++ code")
    parser.add_argument("--python-share", type=float, default=2 / 3,
                        help="the standard library's share of the code, the repository taking the rest (default: 2/3)")
    parser.add_argument("--code-share", type=float, default=0.0, help="share of code (default: 0)")
    parser.add_argument("--multilingual", help="oasst1's train parquet, for chat not in English")
    parser.add_argument("--multilingual-share", type=float, default=0.0, help="share of that chat (default: 0)")
    parser.add_argument("--held-out-code", help="also write code from files the calibration text does not hold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-tool", help="also write tool-call conversations the calibration text does not hold")
    parser.add_argument("--held-out-chat", help="also write chat the calibration text does not hold")
    parser.add_argument("--held-out-tokens", type=int, default=48_000,
                        help="tokens in each held-out file (default: 48000)")
    build(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
