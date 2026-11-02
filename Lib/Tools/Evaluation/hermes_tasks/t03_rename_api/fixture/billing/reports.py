from decimal import Decimal

from billing.invoice import calc_total


def monthly_summary(invoices, tax_rate):
    """Total billed across a month's invoices, each a list of lines with an optional discount."""
    billed = Decimal("0")

    for lines, discount in invoices:
        billed += calc_total(lines, tax_rate, discount)

    return billed
