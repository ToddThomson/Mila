"""Write an arm's Hub benchmark run as the model repo's evaluation result.

A model repo reports a score against a registered benchmark in .eval_results/<task>.yaml; the Hub
shows it on the model page and in that benchmark's leaderboard. This writes the file from a run of
run_inspect.py, so the benchmark, its task, the revision of its definition and the settings come
from the run's own record rather than from whoever types them.
"""

import argparse
import json
import pathlib
import sys


def read_run(arm_directory):
    """The run's environment record, and the inspect-ai log it wrote."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    directory = arm_directory / "inspect"
    environment_path = directory / "environment.json"

    if not environment_path.exists():
        sys.exit(f"{directory} holds no run of run_inspect.py.")

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    logs = list_eval_logs(str(directory))

    if len(logs) != 1:
        sys.exit(f"{directory} holds {len(logs)} inspect-ai logs; a run of run_inspect.py writes one.")

    log = read_eval_log(logs[0], header_only=True)

    if log.status != "success":
        sys.exit(f"The run ended {log.status}: {log.error.message if log.error else 'no error recorded'}.")

    return environment, log


def headline(log, metric):
    """The task's one score: the metric of its one scorer."""
    scores = log.results.scores

    if len(scores) != 1:
        names = ", ".join(score.name for score in scores)
        sys.exit(f"The run has several scorers ({names}); the Hub takes one value per task.")

    metrics = scores[0].metrics

    if metric not in metrics:
        sys.exit(f"The scorer reports {', '.join(metrics)}, not {metric}; choose one with --metric.")

    return metrics[metric].value


def describe(environment, notes):
    """The settings a reader needs to compare this score with another, in their words."""
    parts = [
        f"Mila {environment['mila_version']}",
        f"inspect-ai {environment['inspect_ai']}",
        "greedy decoding",
        f"replies up to {environment['max_tokens']} tokens",
    ]

    if environment["epochs"] > 1:
        parts.append(f"{environment['epochs']} epochs")

    if notes:
        parts.append(notes)

    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm", type=pathlib.Path, help="the arm's directory, e.g. <run>/mila")
    parser.add_argument("--model-repo", required=True, type=pathlib.Path,
                        help="a local clone of the model's Hub repo; the file goes in its .eval_results/")
    parser.add_argument("--metric", default="accuracy", help="the scorer's metric to report")
    parser.add_argument("--notes", help="anything else a reader needs, e.g. the weight format")
    parser.add_argument("--source-url", help="where the run's logs can be read")
    parser.add_argument("--source-name", default="Evaluation logs")
    parser.add_argument("--source-private", action="store_true",
                        help="the source is readable only by people who accepted a gated benchmark's terms")
    arguments = parser.parse_args()

    import yaml

    environment, log = read_run(arguments.arm)

    if environment["arm"] != "mila":
        sys.exit(f"This is the {environment['arm']} arm. The file reports the model its repo holds, which is the mila arm's.")

    if environment["limit"]:
        sys.exit(f"The run took the first {environment['limit']} samples: it checks the setup, and is never a result.")

    # The logs hold every question in plain text, which a gated benchmark's terms forbid publishing.
    if arguments.source_url and environment["gated"] and not arguments.source_private:
        sys.exit(f"{environment['benchmark']} is gated and its logs quote it. Link only a source restricted to people "
                 "who accepted its terms, and say so with --source-private.")

    value = headline(log, arguments.metric)
    entry = {
        "dataset": {
            "id": environment["benchmark"],
            "task_id": environment["task"],
            "revision": environment["revision"],
        },
        # A percentage, as the leaderboards the Hub shows are written.
        "value": round(100 * value, 2),
        "date": log.eval.created,
        "notes": describe(environment, arguments.notes),
    }

    if arguments.source_url:
        entry["source"] = {"url": arguments.source_url, "name": arguments.source_name}

    path = arguments.model_repo / ".eval_results" / f"{environment['task']}.yaml"

    if path.exists():
        sys.exit(f"{path} exists. A result is replaced deliberately: delete the file first.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump([entry], sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"{path}: {entry['value']} on {environment['benchmark']} {environment['task']}")


if __name__ == "__main__":
    main()
