"""Research: an answer reached by combining limits from three spec documents, with a superseded version beside them."""

from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, not_blocked, read_answer

PROMPT = (
    "spec/ is the Wren Transport Protocol specification. Under WTP/2, what is the largest application payload, in "
    "bytes, that a single frame can carry when the frame is encrypted and has the extended header? Write just the "
    "number to answer.txt."
)

# 16384 - 12 (base header) - 8 (extended header) - 8 (nonce) - 16 (tag) = 16340 bytes for the ciphertext; the largest
# whole number of blocks is 16336, and at least one byte of it is padding.
ANSWER = "16335"


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)


def solve(workdir: Path) -> str:
    (workdir / "answer.txt").write_text(ANSWER + "\n")

    return ANSWER


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    answer = read_answer(workdir, "answer.txt")

    if answer is None:
        return False, "no answer.txt"

    return answer.replace(",", "").removesuffix("bytes").strip() == ANSWER, f"answer.txt: {answer[:40]}"


TASK = Task("t05_protocol_arithmetic", "research", PROMPT, build, solve, check)
