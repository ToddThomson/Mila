"""Run one arm of a paired evaluation: the same tasks, prompts and settings on one engine.

An arm is an engine serving a model. `hf` is HuggingFace transformers, the reference; `mila`
is a running MIS. Every setting that could move a score is fixed here and identical across
arms, so that two arms compared by compare_arms.py differ in the engine and nothing else.
In particular the mila arm sends token ids, rendered and tokenized by the same HuggingFace
tokenizer the hf arm uses -- the prompt template and the tokenizer are out of the comparison.
"""

import argparse
import datetime
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_TASKS = "ifeval,gsm8k_cot_llama"
DEFAULT_REFERENCE = "meta-llama/Llama-3.2-3B-Instruct"

# Long enough for every prompt and reply in the default tasks (IFEval replies run to 1,280
# tokens), and the same on both arms: lm-eval left-truncates a context past it.
MAX_LENGTH = 8192

SEED = 1234
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def lm_eval_arguments(arm, reference, url, served_name, tasks, include_path, limit, device, output):
    """The lm-eval command line. Everything outside model_args is shared by both arms."""
    if arm == "hf":
        model = "hf"
        model_args = f"pretrained={reference},dtype=bfloat16,max_length={MAX_LENGTH}"
    else:
        model = "local-completions"
        model_args = (
            f"model={served_name},base_url={url}/v1/completions,"
            f"tokenizer={reference},tokenizer_backend=huggingface,tokenized_requests=True,"
            f"max_length={MAX_LENGTH},num_concurrent=1,timeout=900"
        )

    arguments = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", model,
        "--model_args", model_args,
        "--tasks", tasks,
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        # One prompt at a time on both arms. A batch pads its prompts, and padding changes a
        # BF16 forward pass enough to change a greedy reply.
        "--batch_size", "1",
        "--seed", str(SEED),
        "--log_samples",
        "--output_path", str(output),
    ]

    if arm == "hf":
        arguments += ["--device", device]

    if include_path:
        arguments += ["--include_path", str(include_path)]

    if limit:
        arguments += ["--limit", str(limit)]

    return arguments


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def preflight_mila(url, reference):
    """
    Confirm MIS is up, name what it serves, and prove it takes a token-id prompt as the model's
    input. A server that predates token prompts would encode the list as text, or fail, and
    the arm would measure a different prompt from the hf arm's without saying so.
    """
    import transformers

    try:
        models = request_json(f"{url}/v1/models")
    except urllib.error.URLError as error:
        sys.exit(f"MIS is not answering at {url} ({error.reason}). Start it with MILA_PROTOCOL=openai.")

    served = models["data"][0]
    tokenizer = transformers.AutoTokenizer.from_pretrained(reference)
    # Rendered then encoded without added specials, as lm-eval prepares every request: the
    # template already opens with the BOS token.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one word."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    probe = tokenizer.encode(rendered, add_special_tokens=False)
    payload = {"model": served["id"], "prompt": [probe], "max_tokens": 4, "temperature": 0}

    try:
        reply = request_json(f"{url}/v1/completions", payload)
    except urllib.error.HTTPError as error:
        sys.exit(f"MIS refused a token-id prompt ({error.code}: {error.read().decode()}). "
                 "It needs the Completions token-prompt support this tool depends on.")

    prompt_tokens = reply["usage"]["prompt_tokens"]

    if prompt_tokens != len(probe):
        sys.exit(f"MIS read a {len(probe)}-token prompt as {prompt_tokens} tokens: it encoded the ids "
                 "as text. It needs the Completions token-prompt support this tool depends on.")

    return served


def environment(arm, reference, served, device):
    """What produced the arm's numbers, written beside them."""
    record = {
        "arm": arm,
        "reference": reference,
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mila_version": (REPOSITORY_ROOT / "Version.txt").read_text().strip(),
        "lm_eval": importlib.metadata.version("lm_eval"),
        "transformers": importlib.metadata.version("transformers"),
        "max_length": MAX_LENGTH,
        "seed": SEED,
    }

    if arm == "hf":
        import torch

        record["torch"] = torch.__version__
        record["device"] = torch.cuda.get_device_name(device) if device.startswith("cuda") else device
    else:
        record["served"] = served

    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", choices=["hf", "mila"])
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="the run's directory; this arm writes to <output>/<arm>")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="HuggingFace model id: the hf arm's weights, and both arms' tokenizer and template")
    parser.add_argument("--url", default="http://localhost:8000", help="MIS, for the mila arm")
    parser.add_argument("--device", default="cuda:0", help="the hf arm's device")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--include-path", type=pathlib.Path, help="a directory of task definitions of your own")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N documents of each task; for a smoke run, never for a result")
    arguments = parser.parse_args()

    output = arguments.output / arguments.arm

    # Samples, not any file: an arm that failed before lm-eval wrote them can be run again in place.
    if any(output.glob("**/samples_*.jsonl")):
        sys.exit(f"{output} already holds a run. Choose another --output; compare_arms.py reads whatever is there.")

    output.mkdir(parents=True, exist_ok=True)
    served = None

    if arguments.arm == "mila":
        served = preflight_mila(arguments.url, arguments.reference)
        print(f"MIS serves {served['id']}")

    record = environment(arguments.arm, arguments.reference, served, arguments.device)
    record["tasks"] = arguments.tasks
    record["limit"] = arguments.limit
    (output / "environment.json").write_text(json.dumps(record, indent=2) + "\n")

    served_name = served["id"] if served else None
    command = lm_eval_arguments(arguments.arm, arguments.reference, arguments.url, served_name, arguments.tasks,
                                arguments.include_path, arguments.limit, arguments.device, output)
    print(" ".join(command), flush=True)

    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
