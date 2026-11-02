"""python -m billing.cli QTY:PRICE [QTY:PRICE ...] -- prints the total with 20% tax."""

import sys
from decimal import Decimal

from billing import calc_total
from billing.invoice import Line


def main(arguments):
    lines = []

    for argument in arguments:
        quantity, price = argument.split(":")
        lines.append(Line("item", int(quantity), Decimal(price)))

    print(calc_total(lines, Decimal("0.20")))


if __name__ == "__main__":
    main(sys.argv[1:])
