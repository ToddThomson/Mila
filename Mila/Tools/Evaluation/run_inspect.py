"""Run one arm of a Hugging Face Hub benchmark: the benchmark's own definition, on one engine.

A benchmark registered on the Hub carries an eval.yaml that fixes its prompt, solver and scorer,
and the scores model repos report against it are aggregated into that benchmark's leaderboard.
Every registered language benchmark so far runs on inspect-ai, not lm-eval, so a score that sits
beside the others on such a leaderboard has to come from inspect-ai running that eval.yaml. This
runs it, pinned to one revision of the definition, greedily and one request at a time.

Unlike run_arm.py, the arms are not sent the same token ids: inspect-ai sends chat messages, and
each engine renders them in its own template. So two arms differ in the engine and its template,
which is what a user of either meets. eval_results.py turns an arm's run into the Hub's file.
"""

import argparse
import os
import pathlib
import subprocess
import sys

import servers

DEFAULT_BENCHMARK = "TIGER-Lab/MMLU-Pro"
DEFAULT_REFERENCE = "meta-llama/Llama-3.2-3B-Instruct"

# The same on every arm: each engine has its own default (MIS's is 1,024), and a reply cut short
# scores differently from the same reply finished.
DEFAULT_MAX_TOKENS = 2048

SEED = 1234
SERVED_ARMS = ("mila", "llamacpp")

# inspect-ai's OpenAI-compatible provider reads <SERVICE>_BASE_URL and <SERVICE>_API_KEY.
SERVICES = {
    "mila": "mis",
    "llamacpp": "llamacpp",
}

# A scorer that asks a model whether the answer is right. With no grader bound, inspect-ai grades
# with the model being evaluated, so a 3B model would mark its own work.
JUDGE_SCORERS = ("model_graded_qa", "model_graded_fact")


def read_benchmark(benchmark):
    """The benchmark's current revision, and its eval.yaml at that revision."""
    import huggingface_hub
    import yaml

    api = huggingface_hub.HfApi()

    try:
        info = api.dataset_info(benchmark)
        path = huggingface_hub.hf_hub_download(benchmark, "eval.yaml", repo_type="dataset", revision=info.sha)
    except huggingface_hub.errors.GatedRepoError:
        sys.exit(f"{benchmark} is gated. Accept its terms on its Hub page with the account `hf auth login` uses.")
    except huggingface_hub.errors.EntryNotFoundError:
        sys.exit(f"{benchmark} has no eval.yaml, so it is not a registered benchmark.")

    definition = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))

    return info, definition


def select_task(benchmark, definition, task_id):
    """The task to run, after refusing one this harness cannot run or would not report."""
    framework = definition.get("evaluation_framework")

    if framework != "inspect-ai":
        sys.exit(f"{benchmark} runs on {framework}, not inspect-ai.")

    tasks = definition.get("tasks") or []

    if task_id:
        tasks = [task for task in tasks if task.get("id") == task_id]
    elif len(tasks) > 1:
        names = ", ".join(str(task.get("id")) for task in tasks)
        sys.exit(f"{benchmark} defines several tasks ({names}); choose one with --task.")

    if not tasks:
        sys.exit(f"{benchmark} defines no task {task_id}.")

    task = tasks[0]
    judges = [scorer["name"] for scorer in task.get("scorers", []) if scorer["name"] in JUDGE_SCORERS]

    if judges:
        sys.exit(f"{benchmark}'s {task.get('id')} is scored by a model ({', '.join(judges)}). A judge is a third "
                 "engine in the comparison; ModelEval.md leaves such benchmarks out.")

    return task


def task_spec(benchmark, task, multiple, revision):
    """inspect-ai's name for one task of a Hub benchmark, pinned to a revision."""
    name = f"hf/{benchmark}/{task['id']}" if multiple else f"hf/{benchmark}"

    return f"{name}@{revision}"


def model_arguments(arm, arguments, served_name):
    """The model inspect-ai talks to, and the environment it reads the server from."""
    if arm == "hf":
        model = f"hf/{arguments.reference}"
        native = [f"device={arguments.device}", "dtype=bfloat16", "batch_size=1"]

        return model, native, {}

    service = SERVICES[arm]
    variables = {
        f"{service.upper()}_BASE_URL": f"{arguments.url}/v1",
        f"{service.upper()}_API_KEY": "EMPTY",
    }

    return f"openai-api/{service}/{served_name}", [], variables


def inspect_arguments(arguments, spec, model, native, log_directory):
    """The inspect-ai command line. Everything but the model is shared by every arm."""
    command = [
        sys.executable, "-m", "inspect_ai", "eval", spec,
        "--model", model,
        "--temperature", "0",
        "--max-tokens", str(arguments.max_tokens),
        # One request in flight on every arm. A batch pads its prompts, and padding changes a
        # BF16 forward pass enough to change a greedy reply.
        "--max-connections", "1",
        "--seed", str(SEED),
        "--log-dir", str(log_directory),
        "--display", "plain",
    ]

    for argument in native:
        command += ["-M", argument]

    if arguments.limit:
        command += ["--limit", str(arguments.limit)]

    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", choices=["hf", *SERVED_ARMS])
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="the run's directory; this arm writes to <output>/<arm>/inspect")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="the benchmark's Hub dataset id")
    parser.add_argument("--task", help="the task id in the benchmark's eval.yaml, when it defines several")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="HuggingFace model id: the hf arm's weights, and the served arms' preflight tokenizer")
    parser.add_argument("--url", default="http://localhost:8000", help="the server, for a served arm")
    parser.add_argument("--device", default="cuda:0", help="the hf arm's device")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="the longest reply, on every arm")
    parser.add_argument("--limit", type=int, default=0,
                        help="first N samples; for a smoke run, never for a result")
    arguments = parser.parse_args()

    output = arguments.output / arguments.arm / "inspect"

    if any(output.glob("*.eval")):
        sys.exit(f"{output} already holds a run. Choose another --output; compare_arms.py reads whatever is there.")

    info, definition = read_benchmark(arguments.benchmark)
    task = select_task(arguments.benchmark, definition, arguments.task)
    multiple = len(definition["tasks"]) > 1
    spec = task_spec(arguments.benchmark, task, multiple, info.sha)

    output.mkdir(parents=True, exist_ok=True)
    served = None

    if arguments.arm in SERVED_ARMS:
        served = servers.preflight(arguments.arm, arguments.url, arguments.reference)
        print(f"{servers.SERVER_NAMES[arguments.arm]} serves {served['model']['id']}")

    settings = {
        "benchmark": arguments.benchmark,
        "revision": info.sha,
        "gated": bool(info.gated),
        "task": task.get("id"),
        "epochs": task.get("epochs", 1),
        "scorers": [scorer["name"] for scorer in task["scorers"]],
        "reference": arguments.reference,
        "max_tokens": arguments.max_tokens,
        "temperature": 0.0,
        "limit": arguments.limit,
        "seed": SEED,
    }
    record = servers.environment(arguments.arm, "inspect_ai", served, settings)
    servers.write_environment(output, record)

    served_name = served["model"]["id"] if served else None
    model, native, variables = model_arguments(arguments.arm, arguments, served_name)
    command = inspect_arguments(arguments, spec, model, native, output)
    print(" ".join(command), flush=True)

    sys.exit(subprocess.call(command, env={**os.environ, **variables}))


if __name__ == "__main__":
    main()
