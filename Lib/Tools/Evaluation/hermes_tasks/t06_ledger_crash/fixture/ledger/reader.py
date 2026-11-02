import csv
from decimal import Decimal
from pathlib import Path


def read_entries(directory):
    """Every entry in the CSV exports under `directory`, as (category, amount)."""
    entries = []

    for path in sorted(Path(directory).glob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                entries.append((row["category"].strip().lower(), Decimal(row["amount"])))

    return entries
