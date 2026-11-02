"""
Terminal and ops: configure a tool from its README when the example config predates a rename, then run it. The check
reads the configuration through the tool's own loader and the archive it produced.
"""

import configparser
import fnmatch
import sys
from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, not_blocked, run

PROMPT = (
    "Set syncer up for this directory: it should archive the files in inbox/ into archive/, never copy the "
    "editor's temporary files (their names end in .tmp) or its lock files (their names start with ~$), and keep "
    "the current version of each file plus two older ones. Then run it once."
)

INBOX = {
    "report.txt": "Q3 report, draft 4\n",
    "budget.xlsx": "budget figures\n",
    "notes.md": "# Notes\n",
    "tmp_notes.txt": "not a temporary file, despite the name\n",
    "draft.tmp": "autosave\n",
    "~$budget.xlsx": "lock\n",
}

ARCHIVED = {"report.txt", "budget.xlsx", "notes.md", "tmp_notes.txt"}


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)
    inbox = workdir / "inbox"
    inbox.mkdir()

    for name, text in INBOX.items():
        (inbox / name).write_text(text)


def solve(workdir: Path) -> str:
    (workdir / "syncer.ini").write_text(
        "[sync]\nsource = inbox\ndestination = archive\nexclude = *.tmp, ~$*\nkeep_versions = 3\n")
    run([sys.executable, "syncer.py"], workdir)

    return "Configured and ran syncer."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    validated = run([sys.executable, "syncer.py", "--check"], workdir)

    if validated.returncode != 0:
        return False, f"--check: {(validated.stderr or validated.stdout).strip()[:100]}"

    parser = configparser.ConfigParser()
    parser.read(workdir / "syncer.ini")
    section = parser["sync"]

    if (workdir / section["source"]).resolve() != (workdir / "inbox").resolve():
        return False, f"source is {section['source']}"

    if (workdir / section["destination"]).resolve() != (workdir / "archive").resolve():
        return False, f"destination is {section['destination']}"

    patterns = [pattern.strip() for pattern in section.get("exclude", "").split(",") if pattern.strip()]
    excluded = {name for name in INBOX if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)}

    if excluded != set(INBOX) - ARCHIVED:
        return False, f"exclude {patterns} leaves out {sorted(excluded)}"

    if section.get("keep_versions") != "3":
        return False, f"keep_versions is {section.get('keep_versions')}"

    archive = workdir / "archive"
    archived = {path.name for path in archive.iterdir()} if archive.is_dir() else set()

    return archived == ARCHIVED, f"archive holds {sorted(archived)}"


TASK = Task("t11_configure_syncer", "ops", PROMPT, build, solve, check)
