"""
Debugging: a crash on one supplier's export, in another encoding. Ignoring decode errors, or reading every file as
Latin-1, still runs and gets names wrong; the check reads the names.
"""

import sys
import tempfile
from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, not_blocked, run, tree_digests

PROMPT = (
    "`python3 report.py` crashes since Kessler GmbH started sending their contact list; the traceback is in "
    "error.log. Make the report work again, with every name exactly as the supplier wrote it. The files in vendors/ "
    "are replaced by each night's exports, so the fix must be in the code."
)

# Bytes matter here, so the exports are written by build() rather than kept as fixture files.
EXPORTS = {
    "acme.csv": ("company,contact,email\nAcme Ltd,Zoë Ångström,zoe@acme.example\n"
                 "Acme Ltd,Ravi Shankar,ravi@acme.example\n", "utf-8"),
    "bonnard.csv": ("company,contact,email\nBonnard SA,François Lefèvre,f.lefevre@bonnard.example\n", "utf-8"),
    "kessler.csv": ("company,contact,email\r\nKessler GmbH,Jürgen Müller,jm@kessler.example\r\n"
                    "Kessler GmbH,Siobhán O’Brien,sob@kessler.example\r\n", "cp1252"),
}

README = "Contact exports from our suppliers, one CSV each. Kessler's comes from their old Windows ERP system.\n"

ERROR_LOG = """$ python3 report.py
Traceback (most recent call last):
  File "/srv/vendors/report.py", line 25, in <module>
    main()
    ~~~~^^
  File "/srv/vendors/report.py", line 20, in main
    lines = [f"{company} | {contact} | {email}" for company, contact, email in contacts()]
                                                                               ~~~~~~~~^^
  File "/srv/vendors/report.py", line 12, in contacts
    for row in csv.DictReader(handle):
               ~~~~~~~~~~~~~~^^^^^^^^
  File "/usr/lib/python3.14/csv.py", line 183, in __next__
    self.fieldnames
  File "<frozen codecs>", line 325, in decode
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfc in position 37: invalid start byte
"""

EXPECTED = (
    "Acme Ltd | Ravi Shankar | ravi@acme.example\n"
    "Acme Ltd | Zoë Ångström | zoe@acme.example\n"
    "Bonnard SA | François Lefèvre | f.lefevre@bonnard.example\n"
    "Kessler GmbH | Jürgen Müller | jm@kessler.example\n"
    "Kessler GmbH | Siobhán O’Brien | sob@kessler.example\n"
)

FIX = '''        raw = path.read_bytes()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")

        for row in csv.DictReader(io.StringIO(text, newline="")):
            rows.append((row["company"], row["contact"], row["email"]))
'''


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)
    vendors = workdir / "vendors"
    vendors.mkdir()

    for name, (text, encoding) in EXPORTS.items():
        (vendors / name).write_bytes(text.encode(encoding))

    (vendors / "README.txt").write_text(README)
    (workdir / "error.log").write_text(ERROR_LOG)


def solve(workdir: Path) -> str:
    report = workdir / "report.py"
    text = report.read_text().replace("import csv\n", "import csv\nimport io\n")
    text = text.replace(
        '        with path.open(newline="", encoding="utf-8") as handle:\n'
        '            for row in csv.DictReader(handle):\n'
        '                rows.append((row["company"], row["contact"], row["email"]))\n',
        FIX)
    report.write_text(text)

    return "Kessler's export is Windows-1252; the reader falls back to it when a file is not UTF-8."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    with tempfile.TemporaryDirectory() as scratch:
        build(Path(scratch))
        pristine = tree_digests(Path(scratch) / "vendors")

    if tree_digests(workdir / "vendors") != pristine:
        return False, "vendors/ was edited"

    (workdir / "report.txt").unlink(missing_ok=True)
    result = run([sys.executable, "report.py"], workdir)

    if result.returncode != 0:
        return False, f"exit {result.returncode}: {result.stderr.strip().splitlines()[-1:]}"

    report = (workdir / "report.txt").read_text(encoding="utf-8", errors="replace")

    return report == EXPECTED, f"report: {report.strip()[-80:]!r}"


TASK = Task("t07_vendor_encoding", "debugging", PROMPT, build, solve, check)
