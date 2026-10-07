"""What every served arm checks before it runs, and records beside its results.

A served arm is an engine behind OpenAI's Completions route: MIS, llama-server, or
reference_server.py. Each must read a token-id prompt as the model's input, unchanged. A server
that encoded the list as text, or added a BOS token of its own, would measure a different prompt
from the other arms' without saying so, so the arm refuses to start until a probe proves it.
"""

import datetime
import importlib.metadata
import json
import pathlib
import sys
import urllib.error
import urllib.request

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]

SERVER_NAMES = {
    "mila": "MIS",
    "llamacpp": "llama-server",
    "hf": "the reference server",
}


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def probe_ids(tokenizer_name):
    """A one-turn chat, rendered then encoded without added specials, as lm-eval and BFCL prepare a request."""
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one word."}],
        add_generation_prompt=True,
        tokenize=False,
    )

    return tokenizer.encode(rendered, add_special_tokens=False)


def preflight(arm, url, tokenizer_name):
    """
    Confirm the server answers, name what it serves, and prove it reads token ids as sent.
    Returns what the server says about itself, for the arm's environment record.
    """
    server = SERVER_NAMES[arm]

    try:
        models = request_json(f"{url}/v1/models")
    except urllib.error.URLError as error:
        sys.exit(f"{server} is not answering at {url} ({error.reason}).")

    served = {"model": models["data"][0]}
    probe = probe_ids(tokenizer_name)
    payload = {"model": served["model"]["id"], "prompt": [probe], "max_tokens": 4, "temperature": 0}

    try:
        reply = request_json(f"{url}/v1/completions", payload)
    except urllib.error.HTTPError as error:
        sys.exit(f"{server} refused a token-id prompt ({error.code}: {error.read().decode()}).")

    prompt_tokens = reply["usage"]["prompt_tokens"]

    if prompt_tokens != len(probe):
        sys.exit(f"{server} read a {len(probe)}-token prompt as {prompt_tokens} tokens: it did not take the ids "
                 "as sent (encoded them as text, or added a BOS token of its own).")

    if arm == "llamacpp":
        served["props"] = llama_server_props(url)

    return served


def llama_server_props(url):
    """
    llama-server's build, weights file and settings. Its slot count matters: more than one slot
    lets it batch requests, which the arms never send concurrently but a second client could.
    """
    try:
        props = request_json(f"{url}/props")
    except urllib.error.URLError:
        return {}

    if props.get("total_slots", 1) > 1:
        print(f"llama-server runs {props['total_slots']} slots; start it with -np 1 so no request is batched with another.")

    return props


def environment(arm, harness, served, settings):
    """What produced an arm's numbers, written beside them as environment.json."""
    record = {
        "arm": arm,
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mila_version": (REPOSITORY_ROOT / "Version.txt").read_text().strip(),
        harness: importlib.metadata.version(harness),
        "transformers": importlib.metadata.version("transformers"),
        **settings,
    }

    if served:
        record["served"] = served

    return record


def write_environment(output, record):
    (output / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
