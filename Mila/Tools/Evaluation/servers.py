"""What every served arm checks before it runs, and records beside its results.

A served arm is an engine behind OpenAI's Completions route: MIS, llama-server, or
reference_server.py. Each must read a token-id prompt as the model's input, unchanged. A server
that encoded the list as text, or added a BOS token of its own, would measure a different prompt
from the other arms' without saying so, so the arm refuses to start until a probe proves it.
"""

import datetime
import importlib.metadata
import json
import os
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


def harness_environment(variables=None):
    """
    The environment a harness runs in: UTF-8 for its console, which on Windows is cp1252 otherwise, and
    lm-eval's summary table holds characters cp1252 cannot write -- it fails after the run, results saved.
    """
    return {**os.environ, "PYTHONUTF8": "1", **(variables or {})}


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


def torch_device_record(device):
    """
    A torch device in the form MIS reports its own: the card by name, capability and PCI address.
    The PCI address is what identifies it, since torch's index, like MIS's, is CUDA's and not nvidia-smi's.
    """
    import torch

    if not device.startswith("cuda"):
        return {"name": device}

    properties = torch.cuda.get_device_properties(device)
    record = {
        "index": torch.device(device).index or 0,
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }

    if hasattr(properties, "pci_bus_id"):
        record["pci_bus_id"] = (f"{properties.pci_domain_id:08x}:{properties.pci_bus_id:02x}:"
                                f"{properties.pci_device_id:02x}.0")

    return record


def environment(arm, harness, served, settings, device=None):
    """
    What produced an arm's numbers, written beside them as environment.json. `device` is the card the
    hf arm ran on; a served arm's is the one its server reports. `mila_version` is the library that served
    the mila arm, read from MIS, and the checkout's for the others.
    """
    repository_version = (REPOSITORY_ROOT / "Version.txt").read_text().strip()
    card = served["model"] if served else {}
    record = {
        "arm": arm,
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mila_version": card.get("mila_version") or repository_version,
        "repository_version": repository_version,
        "device": device or card.get("device"),
        harness: importlib.metadata.version(harness),
        "transformers": importlib.metadata.version("transformers"),
        **settings,
    }

    if served:
        record["served"] = served

    return record


def write_environment(output, record):
    (output / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
