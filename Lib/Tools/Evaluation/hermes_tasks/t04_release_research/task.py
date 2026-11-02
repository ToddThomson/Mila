"""
Research: a question a frozen document set answers only across three hops -- release note to RFC to minutes --
with the obvious search landing on the wrong release and the RFC naming a target that slipped.
"""

import re
from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, not_blocked, read_answer

PROMPT = (
    "The docs/ directory holds Larkspur's release notes, RFCs and maintainer meeting minutes. In which release "
    "could the retry backoff first be configured from an environment variable, and which maintainer approved that "
    "change? Write the answer to answer.txt as `<version>, <maintainer name>`."
)


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)


def solve(workdir: Path) -> str:
    (workdir / "answer.txt").write_text("1.5.0, Ines Okafor\n")

    return "1.5.0, approved by Ines Okafor."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    answer = read_answer(workdir, "answer.txt")

    if answer is None:
        return False, "no answer.txt"

    normalized = answer.lower()
    versions = re.findall(r"\d+\.\d+(?:\.\d+)?", normalized)
    right_version = versions in (["1.5.0"], ["1.5"])
    right_person = "ines okafor" in normalized and not any(
        other in normalized for other in ("wilk", "raman", "mensah"))

    return right_version and right_person, f"answer.txt: {answer[:80]}"


TASK = Task("t04_release_research", "research", PROMPT, build, solve, check, short=True)
