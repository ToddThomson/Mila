"""
The Hermes real-world task set (ModelEval.md Phase 8) against a model the inference server is serving.

Each task starts from a fresh workdir, Hermes Agent runs it once in one-shot mode under a HERMES_HOME made fresh
for the run -- no memory, skills, persona or rules carried in from the user's -- and the task's check scores the
workdir and the agent's final reply. The model is run at its defaults: no sampling knob and no reasoning level is sent.

`--self-test` proves every check before any run counts: it must fail on the untouched workdir and pass on the
task's reference solution. Run inside WSL, where Hermes is installed, with workdirs on the Linux filesystem.

    python3 run_hermes_tasks.py --self-test
    python3 run_hermes_tasks.py --model Qwen3.8-27B-cb2-3-refit --out ~/hermes-runs/refit --short
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hermes_tasks.common import PREAMBLE, Task  # noqa: E402

TASKS_DIRECTORY = Path(__file__).resolve().parent / "hermes_tasks"

HERMES_CONFIG = """model:
  default: {model}
  provider: custom
  base_url: {base_url}
  api_key: mila
  api_mode: codex_responses
agent:
  verify_on_stop: false
  max_turns: {max_turns}
"""


def all_tasks() -> list[Task]:
    tasks = []

    for directory in sorted(TASKS_DIRECTORY.iterdir()):
        if directory.is_dir() and (directory / "task.py").is_file():
            tasks.append(importlib.import_module(f"hermes_tasks.{directory.name}.task").TASK)

    return tasks


def chosen(tasks: list[Task], names: list[str] | None, short: bool) -> list[Task]:
    if names:
        unknown = set(names) - {task.name for task in tasks}

        if unknown:
            raise SystemExit(f"no such task: {', '.join(sorted(unknown))}")

        return [task for task in tasks if task.name in names]

    return [task for task in tasks if task.short] if short else tasks


def self_test(tasks: list[Task]) -> bool:
    """Every check fails on the task as given and passes on its reference solution."""
    sound = True

    for task in tasks:
        with tempfile.TemporaryDirectory() as scratch:
            untouched = Path(scratch) / "untouched"
            untouched.mkdir()
            task.build(untouched)
            fails_untouched, untouched_detail = task.check(untouched, "")

            solved = Path(scratch) / "solved"
            solved.mkdir()
            task.build(solved)
            reply = task.solve(solved)
            passes_solved, solved_detail = task.check(solved, reply)

        verdict = "ok" if (not fails_untouched and passes_solved) else "BROKEN"
        sound &= verdict == "ok"
        print(f"{verdict:6} {task.name:32} untouched: {fails_untouched} ({untouched_detail[:70]}) | "
              f"solved: {passes_solved} ({solved_detail[:70]})")

    return sound


def final_reply(transcript: Path) -> str:
    """The agent's last answer, from the stream's closing result record."""
    reply = ""

    for line in transcript.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(event, dict) and event.get("type") == "result":
            for key in ("result", "text", "content", "response"):
                if isinstance(event.get(key), str):
                    reply = event[key]
                    break

    return reply


def prepare_home(out: Path, args: argparse.Namespace) -> Path:
    """
    One HERMES_HOME for the run, made fresh: Hermes installs its runtime into a new home, which is too slow to
    repeat per task, and with only the terminal and file toolsets nothing one task leaves there reaches the next.
    """
    home = out / "hermes-home"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(HERMES_CONFIG.format(
        model=args.model, base_url=args.base_url, max_turns=args.max_turns))

    return home


def run_task(task: Task, out: Path, home: Path, args: argparse.Namespace) -> dict:
    root = out / task.name
    shutil.rmtree(root, ignore_errors=True)
    workdir = root / "work"
    workdir.mkdir(parents=True)

    task.build(workdir)
    prompt_file = root / "prompt.txt"
    prompt_file.write_text(PREAMBLE + task.prompt)

    command = [args.hermes, "--yolo", "chat", "--oneshot", "--ignore-rules", "--format", "stream-json",
               "-t", args.toolsets, "--max-turns", str(args.max_turns), "--run-budget", str(args.run_budget),
               "--query-file", str(prompt_file)]
    environment = dict(os.environ, HERMES_HOME=str(home))
    transcript = root / "transcript.jsonl"

    started = time.monotonic()

    with transcript.open("w") as stdout, (root / "stderr.log").open("w") as stderr:
        try:
            completed = subprocess.run(command, cwd=workdir, env=environment, stdout=stdout, stderr=stderr,
                                       timeout=args.run_budget + 120)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"

    elapsed = time.monotonic() - started
    reply = final_reply(transcript)
    (root / "reply.txt").write_text(reply)

    try:
        passed, detail = task.check(workdir, reply)
    except Exception as error:  # a check that cannot read what the agent left is a failed task, not a crash
        passed, detail = False, f"check raised {type(error).__name__}: {error}"

    return {"task": task.name, "category": task.category, "passed": passed, "detail": detail,
            "exit_code": exit_code, "seconds": round(elapsed, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="prove every check, then exit")
    parser.add_argument("--model", help="the store name the server is serving")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--out", type=Path, help="results directory, on the Linux filesystem")
    parser.add_argument("--short", action="store_true", help="one task a category")
    parser.add_argument("--tasks", nargs="*", help="these tasks only")
    parser.add_argument("--run-budget", type=int, default=1200, help="wall-clock seconds a task may take")
    parser.add_argument("--max-turns", type=int, default=90)
    parser.add_argument("--toolsets", default="terminal,file")
    parser.add_argument("--hermes", default=shutil.which("hermes") or "hermes")
    args = parser.parse_args()

    tasks = chosen(all_tasks(), args.tasks, args.short)

    if args.self_test:
        raise SystemExit(0 if self_test(tasks) else 1)

    if not args.model or not args.out:
        parser.error("--model and --out are needed for a run")

    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    home = prepare_home(args.out, args)

    for task in tasks:
        result = run_task(task, args.out, home, args)
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'fail'}  {task.name:32} {result['seconds']:>7.1f} s  "
              f"exit {result['exit_code']}  {result['detail'][:80]}", flush=True)

        with (args.out / "results.jsonl").open("a") as log:
            log.write(json.dumps({"model": args.model, **result}) + "\n")

    passed = sum(result["passed"] for result in results)
    print(f"{passed}/{len(results)} passed, {sum(result['seconds'] for result in results) / 60:.1f} min")


if __name__ == "__main__":
    main()
