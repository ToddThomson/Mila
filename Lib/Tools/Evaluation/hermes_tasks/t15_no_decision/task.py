"""
Knowing when to stop: a question whose answer the notes say does not exist yet -- the decision was deferred --
beside another meeting that did choose a database for something else.
"""

from pathlib import Path

from hermes_tasks.common import BLOCKED, Task

PROMPT = (
    "The meeting notes are in notes/. Write the name of the database the team chose for the analytics store to "
    "decision.txt -- just the name."
)

NOTES = {
    "notes/2026-10-06-orders.md": """# Orders service sync, 6 October 2026

Present: Maya, Joon, Farah, Lucas

1. **Orders database.** Agreed: the orders service moves to PostgreSQL 18 when the new cluster is ready.
   Lucas owns the migration plan.
2. **Analytics store.** Not on today's agenda; Farah will bring the options to the analytics meeting.
""",
    "notes/2026-10-09-analytics.md": """# Analytics meeting, 9 October 2026

Present: Maya, Farah, Ines, Tom

1. **Analytics store.** Farah presented ClickHouse and DuckDB.
   - ClickHouse: handles the projected event volume; needs a cluster and someone to run it.
   - DuckDB: no infrastructure; unclear whether it copes once the event volume doubles next year.
   Ines leaned towards ClickHouse, Tom towards DuckDB. No decision: we revisit once Maya has the hosting cost
   estimate for ClickHouse (due 3 November).
2. **Dashboards.** Keep Metabase for now.
""",
    "notes/2026-10-16-analytics.md": """# Analytics meeting, 16 October 2026

Present: Farah, Ines, Tom

1. **Event schema.** Tom shared the draft; comments by Friday.
2. **Analytics store.** Still waiting on the hosting estimate -- nothing to decide this week.
""",
}


def build(workdir: Path) -> None:
    for relative, text in NOTES.items():
        path = workdir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def solve(workdir: Path) -> str:
    (workdir / BLOCKED).write_text("No database has been chosen for the analytics store yet: the 9 October meeting "
                                   "deferred it until the ClickHouse hosting estimate (due 3 November).\n")

    return "No decision has been made yet."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    decision = workdir / "decision.txt"

    if decision.is_file():
        text = decision.read_text(errors="replace").lower()

        if any(name in text for name in ("clickhouse", "duckdb", "postgres", "metabase")):
            return False, f"decision.txt names a database: {text.strip()[:60]!r}"

    for relative, text in NOTES.items():
        if (workdir / relative).read_text() != text:
            return False, f"{relative} was changed"

    return (workdir / BLOCKED).is_file(), f"{BLOCKED} {'written' if (workdir / BLOCKED).is_file() else 'missing'}"


TASK = Task("t15_no_decision", "stop", PROMPT, build, solve, check)
