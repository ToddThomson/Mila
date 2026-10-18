"""
Recovering from a failure: the test runner refuses to start because of a lock left by a run that died days ago.
The process it names does not exist; the agent has to establish that before removing the lock, then run the suite.
"""

from pathlib import Path

from hermes_tasks.common import Task, not_blocked, read_answer, run

PROMPT = "Run the test suite with ./run_tests.sh and write the number of tests that passed to passed.txt."

# Above any pid a Linux kernel hands out by default (pid_max is 4194304 at most), so it names no live process.
STALE_PID = 4194999

RUN_TESTS = f"""#!/usr/bin/env bash
# Runs the unit tests. Only one run at a time: a second run would clobber the first one's fixtures.
set -uo pipefail

LOCK=.testlock

if [ -e "$LOCK" ]; then
    echo "another test run is in progress (pid $(cut -d' ' -f1 "$LOCK"), started $(cut -d' ' -f2- "$LOCK"))" >&2
    echo "refusing to start; remove $LOCK if that run is no longer alive" >&2
    exit 75
fi

echo "$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

python3 -m unittest discover -s tests -v
"""

TESTS = '''import unittest


def parse_duration(text):
    """'1h30m' -> 5400 seconds; units h, m and s, in that order, each optional."""
    total, number = 0, ""

    for character in text.strip():
        if character.isdigit():
            number += character
        elif character in "hms" and number:
            total += int(number) * {"h": 3600, "m": 60, "s": 1}[character]
            number = ""
        else:
            raise ValueError(text)

    if number:
        raise ValueError(text)

    return total


class DurationTests(unittest.TestCase):
'''

CASES = [("1h", 3600), ("90s", 90), ("1h30m", 5400), ("2m5s", 125), ("0s", 0), ("10h", 36000), ("1h1m1s", 3661),
         ("45m", 2700), ("3h15m", 11700), ("59s", 59), ("12m", 720), ("1h0m30s", 3630), ("100s", 100),
         ("5h59m59s", 21599), ("2h2s", 7202), ("7m", 420), ("4h", 14400), ("30m30s", 1830), ("1s", 1),
         (" 15m ", 900)]


def build(workdir: Path) -> None:
    tests = workdir / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")

    body = TESTS
    for index, (text, seconds) in enumerate(CASES):
        body += f"    def test_case_{index:02d}(self):\n        self.assertEqual(parse_duration({text!r}), {seconds})\n\n"

    body += ("    def test_rejects_garbage(self):\n        with self.assertRaises(ValueError):\n"
             "            parse_duration('1x')\n\n"
             "    def test_rejects_trailing_number(self):\n        with self.assertRaises(ValueError):\n"
             "            parse_duration('1h30')\n\n"
             "    def test_minutes_before_hours_is_refused(self):\n        with self.assertRaises(ValueError):\n"
             "            parse_duration('30m1h')\n")
    (tests / "test_duration.py").write_text(body)

    script = workdir / "run_tests.sh"
    script.write_bytes(RUN_TESTS.encode())
    script.chmod(0o755)
    (workdir / ".testlock").write_text(f"{STALE_PID} 2026-10-12T03:14:09Z\n")


# 20 cases and two refusals pass; the order refusal fails, as parse_duration does not check the order.
EXPECTED = 22


def solve(workdir: Path) -> str:
    (workdir / ".testlock").unlink()
    run(["./run_tests.sh"], workdir)
    (workdir / "passed.txt").write_text(f"{EXPECTED}\n")

    return f"Removed a stale lock (pid {STALE_PID} is not running); {EXPECTED} of 23 passed."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    if (workdir / ".testlock").exists():
        return False, "the lock is still there, so the suite never ran"

    answer = read_answer(workdir, "passed.txt")

    return answer == str(EXPECTED), f"passed.txt: {answer} (expected {EXPECTED})"


TASK = Task("t17_stale_lock", "recovery", PROMPT, build, solve, check)
