import sys
from collections import defaultdict
from decimal import Decimal

from ledger.reader import read_entries


def summarize(directory):
    totals = defaultdict(Decimal)

    for category, amount in read_entries(directory):
        totals[category] += amount

    for category in sorted(totals):
        print(f"{category}: {totals[category]:.2f}")

    print(f"total: {sum(totals.values(), Decimal(0)):.2f}")


def main(arguments):
    if len(arguments) != 2 or arguments[0] != "summarize":
        print("usage: python3 -m ledger summarize DIRECTORY", file=sys.stderr)
        return 2

    summarize(arguments[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
