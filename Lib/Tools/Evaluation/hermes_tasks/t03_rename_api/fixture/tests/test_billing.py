import unittest
from decimal import Decimal

from billing import Line, calc_total, monthly_summary, to_csv_row

MUGS = [Line("mug", 2, Decimal("8.50"))]


class BillingTests(unittest.TestCase):
    def test_total_with_tax(self):
        self.assertEqual(calc_total(MUGS, Decimal("0.20")), Decimal("20.40"))

    def test_discount_before_tax(self):
        self.assertEqual(calc_total(MUGS, Decimal("0.20"), Decimal("0.10")), Decimal("18.36"))

    def test_monthly_summary(self):
        self.assertEqual(monthly_summary([(MUGS, Decimal("0")), (MUGS, Decimal("0.10"))], Decimal("0.20")),
                         Decimal("38.76"))

    def test_csv_row(self):
        self.assertEqual(to_csv_row(7, "Ada", MUGS, Decimal("0.20")), "7,Ada,1,20.40")


if __name__ == "__main__":
    unittest.main()
