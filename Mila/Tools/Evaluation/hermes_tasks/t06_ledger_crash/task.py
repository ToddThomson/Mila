"""
Debugging: a nightly job's traceback, whose cause is a new export format; fixing the error the log shows exposes a
second one the log does not, and the data may not be edited to get round either.
"""

import sys
import tempfile
from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, not_blocked, run, tree_digests

PROMPT = (
    "Last night's summary job failed; its output is in error.log. Fix the code so that "
    "`python3 -m ledger summarize data` works again. The files in data/ are exports from other systems and must "
    "not be edited."
)

EXPECTED = "groceries: 311.50\nrent: 3650.00\ntravel: 1355.60\nutilities: 389.25\ntotal: 5706.35"

READER_FIX = (
    '                row = {key.strip().lower(): value for key, value in row.items()}\n'
    '                entries.append((row["category"].strip().lower(), Decimal(row["amount"].replace(",", ""))))'
)


MARCH = (
    'Date,Category,Amount \n'
    '2026-03-03,Rent,"1,250.00"\n'
    '2026-03-06,Groceries,71.15\n'
    '2026-03-14,Travel,"1,045.60"\n'
    '2026-03-22,Utilities,119.80\n'
)


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)

    # Written here rather than kept as a fixture file: the header's trailing space is part of the fault.
    (workdir / "data" / "2026-03.csv").write_bytes(MARCH.encode())


def pristine_data() -> dict[str, str]:
    with tempfile.TemporaryDirectory() as scratch:
        build(Path(scratch))

        return tree_digests(Path(scratch) / "data")


def solve(workdir: Path) -> str:
    reader = workdir / "ledger" / "reader.py"
    reader.write_text(reader.read_text().replace(
        '                entries.append((row["category"].strip().lower(), Decimal(row["amount"])))', READER_FIX))

    return "Fixed the reader for the new export's headers and thousands separators."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    if tree_digests(workdir / "data") != pristine_data():
        return False, "data/ was edited"

    result = run([sys.executable, "-m", "ledger", "summarize", "data"], workdir)

    if result.returncode != 0:
        return False, f"exit {result.returncode}: {result.stderr.strip().splitlines()[-1:]}"

    return result.stdout.strip() == EXPECTED, f"output: {result.stdout.strip()[:120]!r}"


TASK = Task("t06_ledger_crash", "debugging", PROMPT, build, solve, check, short=True)
