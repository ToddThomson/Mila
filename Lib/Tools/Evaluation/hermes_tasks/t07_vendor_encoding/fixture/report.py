"""Writes report.txt: every vendor contact, by company, then name."""

import csv
from pathlib import Path


def contacts():
    rows = []

    for path in sorted(Path("vendors").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append((row["company"], row["contact"], row["email"]))

    return sorted(rows)


def main():
    lines = [f"{company} | {contact} | {email}" for company, contact, email in contacts()]
    Path("report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
