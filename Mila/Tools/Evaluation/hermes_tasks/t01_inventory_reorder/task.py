"""Coding: add a feature to a small package from the issue that asks for it, with tests."""

from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, fixture_path, not_blocked, run_hidden_tests

PROMPT = "Implement the feature described in ISSUE.md. The existing tests must keep passing."

HIDDEN_TESTS = '''
import unittest

from stockroom import Inventory


def shop(**stock):
    inventory = Inventory()

    for sku, quantity in stock.items():
        inventory.add(sku.replace("_", "-"), quantity)

    return inventory


class ReorderListHiddenTests(unittest.TestCase):
    def test_orders_by_days_of_cover(self):
        inventory = shop(A_1=100, B_2=100, C_3=100)
        inventory.record_sale("A-1", 28, day=5)    # 1/day, 72 left -> 72 days
        inventory.record_sale("B-2", 56, day=6)    # 2/day, 44 left -> 22 days
        inventory.record_sale("C-3", 84, day=7)    # 3/day, 16 left -> 5.33 days

        self.assertEqual(inventory.reorder_list(today=27, threshold_days=30), ["C-3", "B-2"])

    def test_window_is_the_28_days_ending_today(self):
        inventory = shop(A_1=100)
        inventory.record_sale("A-1", 50, day=0)    # outside the window ending on day 28
        inventory.record_sale("A-1", 28, day=1)    # inside: 1/day, 22 left -> 22 days

        self.assertEqual(inventory.reorder_list(today=28, threshold_days=23), ["A-1"])
        self.assertEqual(inventory.reorder_list(today=28, threshold_days=22), [])

    def test_sales_after_today_do_not_count(self):
        inventory = shop(A_1=100)
        inventory.record_sale("A-1", 90, day=40)

        self.assertEqual(inventory.reorder_list(today=30, threshold_days=1000), [])

    def test_no_sales_never_reorders_even_when_empty(self):
        inventory = shop(A_1=1, B_2=10)
        inventory.record_sale("A-1", 1, day=100)   # A-1 sold out long before the window

        self.assertEqual(inventory.reorder_list(today=200, threshold_days=1000), [])

    def test_sold_out_in_window_is_most_urgent(self):
        inventory = shop(A_1=5, B_2=100)
        inventory.record_sale("A-1", 5, day=10)
        inventory.record_sale("B-2", 98, day=10)   # 3.5/day, 2 left -> 0.57 days

        self.assertEqual(inventory.reorder_list(today=20, threshold_days=10), ["A-1", "B-2"])

    def test_ties_come_in_sku_order(self):
        inventory = shop(Z_9=30, A_1=30, M_5=30)

        for sku in ("Z-9", "A-1", "M-5"):
            inventory.record_sale(sku, 14, day=3)  # 0.5/day, 16 left -> 32 days each

        self.assertEqual(inventory.reorder_list(today=10, threshold_days=40), ["A-1", "M-5", "Z-9"])

    def test_threshold_is_strict(self):
        inventory = shop(A_1=56)
        inventory.record_sale("A-1", 28, day=0)    # 1/day, 28 left -> exactly 28 days

        self.assertEqual(inventory.reorder_list(today=0, threshold_days=28), [])
        self.assertEqual(inventory.reorder_list(today=0, threshold_days=28.5), ["A-1"])


if __name__ == "__main__":
    unittest.main()
'''

SOLUTION = '''

    def reorder_list(self, today, threshold_days):
        """SKUs with fewer than `threshold_days` of cover at the last 28 days' average sales, most urgent first."""
        urgent = []

        for sku in self.skus():
            sold = sum(quantity for quantity, day in self.sales(sku) if today - 27 <= day <= today)

            if sold == 0:
                continue

            cover = self._quantity[sku] / (sold / 28)

            if cover < threshold_days:
                urgent.append((cover, sku))

        return [sku for _, sku in sorted(urgent)]
'''

SOLUTION_TEST = '''

class ReorderListTests(unittest.TestCase):
    def test_fast_seller_is_listed(self):
        inventory = Inventory()
        inventory.add("A-1", 10)
        inventory.record_sale("A-1", 8, day=1)

        self.assertEqual(inventory.reorder_list(today=1, threshold_days=30), ["A-1"])
'''


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)


def solve(workdir: Path) -> str:
    inventory = workdir / "stockroom" / "inventory.py"
    inventory.write_text(inventory.read_text() + SOLUTION)

    tests = workdir / "tests" / "test_inventory.py"
    text = tests.read_text().replace('\n\nif __name__ == "__main__":', SOLUTION_TEST + '\n\nif __name__ == "__main__":')
    tests.write_text(text)

    return "Added Inventory.reorder_list with tests."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    added_tests = any("reorder_list" in path.read_text(errors="replace") for path in (workdir / "tests").glob("*.py"))

    if not added_tests:
        return False, "no test calls reorder_list"

    original_tests = (fixture_path(__file__) / "tests" / "test_inventory.py").read_text()

    # The agent's own tests run too, and the original ones as the fixture had them, beside the hidden tests.
    passed, detail = run_hidden_tests(workdir, {"tests/test_reorder_hidden.py": HIDDEN_TESTS,
                                                "tests/test_inventory_original.py": original_tests})

    return passed, detail


TASK = Task("t01_inventory_reorder", "coding", PROMPT, build, solve, check, short=True)
