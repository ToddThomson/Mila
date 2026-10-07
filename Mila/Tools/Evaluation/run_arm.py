"""Run one arm of a paired evaluation: the same tasks, prompts and settings on one engine.

An arm is an engine serving a model. `hf` is HuggingFace transformers, the reference; `mila` is a
running MIS; `llamacpp` is a running llama-server. Every setting that could move a score is fixed
here and identical across arms, so that two arms compared by compare_arms.py differ in the engine
and nothing else. In particular the served arms are sent token ids, rendered and tokenized by the
same HuggingFace tokenizer the hf arm uses -- the prompt template and the tokenizer are out of the
comparison.
"""

import argparse
import json
import pathlib
import subprocess
import sys

import servers

DEFAULT_TASKS = "ifeval,gsm8k_cot_llama"
DEFAULT_REFERENCE = "meta-llama/Llama-3.2-3B-Instruct"

# Long enough for every prompt and reply in the default tasks (IFEval replies run to 1,280
# tokens). lm-eval left-truncates a context past it, identically on every arm, so a long-context
# task must raise it past its largest band.
DEFAULT_MAX_LENGTH = 8192

SEED = 1234
SERVED_ARMS = ("mila", "llamacpp")

# lm-eval 0.4.13's RULER group, and the documents it generates for each band, in band order.
RULER_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot",
)
RULER_SAMPLES_PER_BAND = 500


def ruler_samples(tasks, metadata, per_band):
    """
    lm-eval's --samples selection for the first `per_band` documents of every RULER band. --limit
    cannot do it: RULER lays its bands end to end, so the first N documents are all the shortest band.
    """
    names = []

    for task in tasks.split(","):
        names += RULER_TASKS if task == "ruler" else [task]

    if any(name not in RULER_TASKS for name in names):
        sys.exit("--per-band applies to RULER tasks only; run the others as an arm of their own.")

    bands = json.loads(metadata or "{}").get("max_seq_lengths")

    if not bands:
        sys.exit("--per-band needs the bands: --metadata '{\"max_seq_lengths\": [4096, 8192, ...]}'.")

    if per_band > RULER_SAMPLES_PER_BAND:
        sys.exit(f"RULER generates {RULER_SAMPLES_PER_BAND} documents a band; --per-band cannot exceed it.")

    indices = [band * RULER_SAMPLES_PER_BAND + index for band in range(len(bands)) for index in range(per_band)]

    return json.dumps({name: indices for name in names})


def lm_eval_arguments(arm, arguments, served_name, output):
    """The lm-eval command line. Everything outside model_args is shared by every arm."""
    if arm == "hf":
        model = "hf"
        model_args = f"pretrained={arguments.reference},dtype=bfloat16,max_length={arguments.max_length}"
    else:
        model = "local-completions"
        model_args = (
            f"model={served_name},base_url={arguments.url}/v1/completions,"
            f"tokenizer={arguments.reference},tokenizer_backend=huggingface,tokenized_requests=True,"
            f"max_length={arguments.max_length},num_concurrent=1,timeout=3600"
        )

    command = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", model,
        "--model_args", model_args,
        "--tasks", arguments.tasks,
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        # One prompt at a time on every arm. A batch pads its prompts, and padding changes a
        # BF16 forward pass enough to change a greedy reply.
        "--batch_size", "1",
        "--seed", str(SEED),
        "--log_samples",
        "--output_path", str(output),
    ]

    if arm == "hf":
        command += ["--device", arguments.device]

    if arguments.include_path:
        command += ["--include_path", str(arguments.include_path)]

    if arguments.metadata:
        command += ["--metadata", arguments.metadata]

    if arguments.limit:
        command += ["--limit", str(arguments.limit)]

    if arguments.per_band:
        command += ["--samples", ruler_samples(arguments.tasks, arguments.metadata, arguments.per_band)]

    return command


def hf_device_record(device):
    import torch

    name = torch.cuda.get_device_name(device) if device.startswith("cuda") else device

    return {"torch": torch.__version__, "device": name}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", choices=["hf", *SERVED_ARMS])
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="the run's directory; this arm writes to <output>/<arm>")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="HuggingFace model id: the hf arm's weights, and every arm's tokenizer and template")
    parser.add_argument("--url", default="http://localhost:8000", help="the server, for a served arm")
    parser.add_argument("--device", default="cuda:0", help="the hf arm's device")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--include-path", type=pathlib.Path, help="a directory of task definitions of your own")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="prompt and reply together; for RULER, its largest band plus the reply")
    parser.add_argument("--metadata", help="JSON passed to the tasks, e.g. RULER's {\"max_seq_lengths\": [...]}")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N documents of each task; for a smoke run, never for a result")
    parser.add_argument("--per-band", type=int, default=0,
                        help="RULER only: the first N documents of each band, the same N on every arm")
    arguments = parser.parse_args()

    if arguments.limit and arguments.per_band:
        sys.exit("--limit and --per-band select documents two ways; give one.")

    if arguments.metadata:
        try:
            json.loads(arguments.metadata)
        except json.JSONDecodeError as error:
            sys.exit(f"--metadata is not JSON: {error}")

    output = arguments.output / arguments.arm

    # Samples, not any file: an arm that failed before lm-eval wrote them can be run again in place.
    if any(output.glob("**/samples_*.jsonl")):
        sys.exit(f"{output} already holds a run. Choose another --output; compare_arms.py reads whatever is there.")

    output.mkdir(parents=True, exist_ok=True)
    served = None

    if arguments.arm in SERVED_ARMS:
        served = servers.preflight(arguments.arm, arguments.url, arguments.reference)
        print(f"{servers.SERVER_NAMES[arguments.arm]} serves {served['model']['id']}")

    settings = {
        "reference": arguments.reference,
        "tasks": arguments.tasks,
        "metadata": arguments.metadata,
        "max_length": arguments.max_length,
        "limit": arguments.limit,
        "per_band": arguments.per_band,
        "seed": SEED,
    }

    if arguments.arm == "hf":
        settings.update(hf_device_record(arguments.device))

    record = servers.environment(arguments.arm, "lm_eval", served, settings)
    servers.write_environment(output, record)

    served_name = served["model"]["id"] if served else None
    command = lm_eval_arguments(arguments.arm, arguments, served_name, output)
    print(" ".join(command), flush=True)

    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
