"""Coding: a rename across a package, its tests and its docs, keeping the old name as a deprecated alias."""

import re
from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, fixture_path, not_blocked, run_hidden_tests

PROMPT = (
    "Rename `calc_total` to `compute_invoice_total` throughout this project -- the billing package, its tests and "
    "the README. Code outside this repository still calls `calc_total`, so keep it working as a deprecated alias "
    "that emits a DeprecationWarning when it is called."
)

HIDDEN_TESTS = '''
import subprocess
import sys
import unittest
import warnings
from decimal import Decimal

import billing
from billing import Line, monthly_summary, to_csv_row
from billing.invoice import compute_invoice_total

MUGS = [Line("mug", 2, Decimal("8.50"))]


class RenameHiddenTests(unittest.TestCase):
    def test_new_name_from_both_places(self):
        self.assertIs(billing.compute_invoice_total, compute_invoice_total)
        self.assertEqual(compute_invoice_total(MUGS, Decimal("0.20"), Decimal("0.10")), Decimal("18.36"))

    def test_old_name_warns_and_agrees(self):
        with self.assertWarns(DeprecationWarning):
            total = billing.calc_total(MUGS, Decimal("0.20"), Decimal("0.10"))

        self.assertEqual(total, Decimal("18.36"))

    def test_callers_do_not_go_through_the_alias(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self.assertEqual(monthly_summary([(MUGS, Decimal("0"))], Decimal("0.20")), Decimal("20.40"))
            self.assertEqual(to_csv_row(7, "Ada", MUGS, Decimal("0.20")), "7,Ada,1,20.40")

        cli = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-m", "billing.cli", "2:8.50", "1:3.00"],
                             capture_output=True, text=True)
        self.assertEqual(cli.stdout.strip(), "24.00", cli.stderr)


if __name__ == "__main__":
    unittest.main()
'''

ALIAS = '''

def calc_total(lines, tax_rate, discount=Decimal("0")):
    """Deprecated name of compute_invoice_total."""
    warnings.warn("calc_total is deprecated; use compute_invoice_total", DeprecationWarning, stacklevel=2)

    return compute_invoice_total(lines, tax_rate, discount)
'''


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)


def solve(workdir: Path) -> str:
    for path in [*workdir.rglob("*.py"), workdir / "README.md"]:
        path.write_text(path.read_text().replace("calc_total", "compute_invoice_total"))

    invoice = workdir / "billing" / "invoice.py"
    invoice.write_text("import warnings\n" + invoice.read_text() + ALIAS)

    package = workdir / "billing" / "__init__.py"
    package.write_text(package.read_text()
                       .replace("Line, compute_invoice_total", "Line, calc_total, compute_invoice_total")
                       .replace('"compute_invoice_total",', '"calc_total", "compute_invoice_total",'))

    return "Renamed, with a deprecated alias."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    for caller in ("reports.py", "export.py", "cli.py"):
        if "calc_total" in (workdir / "billing" / caller).read_text():
            return False, f"billing/{caller} still says calc_total"

    for line in (workdir / "README.md").read_text().splitlines():
        if "calc_total" in line and not re.search("deprecat", line, re.IGNORECASE):
            return False, f"README still documents calc_total: {line.strip()[:80]}"

    original_tests = (fixture_path(__file__) / "tests" / "test_billing.py").read_text()

    return run_hidden_tests(workdir, {"tests/test_rename_hidden.py": HIDDEN_TESTS,
                                      "tests/test_billing_original.py": original_tests})


TASK = Task("t03_rename_api", "coding", PROMPT, build, solve, check)
