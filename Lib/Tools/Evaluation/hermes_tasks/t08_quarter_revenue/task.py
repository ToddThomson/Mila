"""
Data work: a quarter's revenue per region from an orders export with three date conventions, three currency
formats, refunds, repeated rows and monthly exchange rates. The export is rendered from known values, so the check
compares against those values rather than against a second parser.
"""

import csv
import io
import json
import random
import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from hermes_tasks.common import Task, not_blocked

PROMPT = (
    "data/orders.csv is the shop's order export and data/README.md explains it. Work out the net revenue for Q3 "
    "2026 (July to September) in each region, in euros, and write it to revenue.json as an object mapping each "
    "region's name in lower case to its total."
)

REGIONS = ("north", "south", "west")

# Euros per unit of each currency, by month.
RATES = {
    "USD": {6: "0.9210", 7: "0.9174", 8: "0.9302", 9: "0.9255", 10: "0.9188"},
    "GBP": {6: "1.1830", 7: "1.1795", 8: "1.1712", 9: "1.1868", 10: "1.1903"},
}

README = """# Order export

`orders.csv` has one row per order or refund, from three shops:

- `eu` -- the European shop. Dates are day first (`DD/MM/YYYY`). Prices are in euros, written the European way:
  `€1.234,50`.
- `us` -- the US shop. Dates are month first (`MM/DD/YYYY`). Prices are in US dollars: `$1,234.50`.
- `web` -- the web shop. Dates are ISO (`YYYY-MM-DD`). Prices are in the customer's currency, written as the
  currency code and a plain number: `GBP 1234.50`.

`type` is `sale` or `refund`. A refund's amount is money paid back, written as a positive number; it comes off
revenue in the month it was paid.

The region column was typed by hand and is not consistent about case or spaces.

The export job sometimes writes an order twice. An `order_id` counts once.

Convert each row to euros at the rate for its month in `fx.csv` (euros per unit of the currency), add up, and
round only the final totals to cents.
"""


def records() -> list[dict]:
    """The orders as values, before the export writes them out; a repeated order appears twice."""
    generator = random.Random(20261018)
    rows = []

    for index in range(220):
        shop = generator.choice(("eu", "us", "web"))
        rows.append({
            "order_id": f"O-{10000 + index}",
            "day": date(2026, 6, 15) + timedelta(days=generator.randrange(123)),
            "shop": shop,
            "currency": {"eu": "EUR", "us": "USD"}.get(shop) or generator.choice(("EUR", "GBP", "USD")),
            "amount": Decimal(generator.randrange(500, 250000)) / 100,
            "type": "refund" if generator.random() < 0.08 else "sale",
            "region": generator.choice(REGIONS),
            "region_written": None,
        })

    for row in rows:
        row["region_written"] = generator.choice(
            (row["region"], row["region"].title(), row["region"].upper(), f" {row['region']} "))

    for _ in range(12):
        repeated = dict(generator.choice(rows))
        rows.insert(generator.randrange(len(rows)), repeated)

    return rows


def written_date(row: dict) -> str:
    day = row["day"]

    if row["shop"] == "eu":
        return day.strftime("%d/%m/%Y")

    if row["shop"] == "us":
        return day.strftime("%m/%d/%Y")

    return day.isoformat()


def written_amount(row: dict) -> str:
    amount = row["amount"]

    if row["shop"] == "eu":
        return "€" + f"{amount:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")

    if row["shop"] == "us":
        return f"${amount:,.2f}"

    return f"{row['currency']} {amount:.2f}"


def truth() -> dict[str, Decimal]:
    totals = {region: Decimal("0") for region in REGIONS}
    seen = set()

    for row in records():
        if row["order_id"] in seen:
            continue

        seen.add(row["order_id"])

        if not date(2026, 7, 1) <= row["day"] <= date(2026, 9, 30):
            continue

        rate = Decimal("1") if row["currency"] == "EUR" else Decimal(RATES[row["currency"]][row["day"].month])
        euros = row["amount"] * rate
        totals[row["region"]] += -euros if row["type"] == "refund" else euros

    return {region: total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) for region, total in totals.items()}


def build(workdir: Path) -> None:
    data = workdir / "data"
    data.mkdir()

    orders = io.StringIO()
    writer = csv.writer(orders, lineterminator="\n")
    writer.writerow(["order_id", "date", "shop", "region", "type", "amount"])

    for row in records():
        writer.writerow([row["order_id"], written_date(row), row["shop"], row["region_written"], row["type"],
                         written_amount(row)])

    (data / "orders.csv").write_text(orders.getvalue(), encoding="utf-8")

    fx = ["month,currency,eur_per_unit"]
    fx += [f"2026-{month:02d},{currency},{rate}" for currency, months in RATES.items() for month, rate in months.items()]
    (data / "fx.csv").write_text("\n".join(fx) + "\n")
    (data / "README.md").write_text(README)


def solve(workdir: Path) -> str:
    """Parses the export as written, so the README is proven sufficient, not only the generator."""
    rates = {}

    with (workdir / "data" / "fx.csv").open() as handle:
        for row in csv.DictReader(handle):
            rates[(row["currency"], int(row["month"][5:]))] = Decimal(row["eur_per_unit"])

    totals = {region: Decimal("0") for region in REGIONS}
    seen = set()

    with (workdir / "data" / "orders.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["order_id"] in seen:
                continue

            seen.add(row["order_id"])
            pattern = {"eu": "%d/%m/%Y", "us": "%m/%d/%Y"}.get(row["shop"], "%Y-%m-%d")
            day = datetime.strptime(row["date"], pattern).date()

            if not date(2026, 7, 1) <= day <= date(2026, 9, 30):
                continue

            written = row["amount"]

            if written.startswith("€"):
                currency, amount = "EUR", Decimal(written[1:].replace(".", "").replace(",", "."))
            elif written.startswith("$"):
                currency, amount = "USD", Decimal(written[1:].replace(",", ""))
            else:
                currency, number = written.split()
                amount = Decimal(number)

            euros = amount * (Decimal("1") if currency == "EUR" else rates[(currency, day.month)])
            totals[row["region"].strip().lower()] += -euros if row["type"] == "refund" else euros

    result = {region: float(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) for region, total in totals.items()}
    (workdir / "revenue.json").write_text(json.dumps(result, indent=2))

    return json.dumps(result)


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    path = workdir / "revenue.json"

    if not path.is_file():
        return False, "no revenue.json"

    try:
        written = json.loads(path.read_text())
        given = {str(key).strip().lower(): Decimal(re.sub(r"[^\d.\-]", "", str(value))) for key, value in written.items()}
    except (ValueError, ArithmeticError, AttributeError) as error:
        return False, f"revenue.json unreadable: {error}"

    expected = truth()

    if set(given) != set(expected):
        return False, f"regions {sorted(given)}"

    off = {region: given[region] - expected[region] for region in expected if abs(given[region] - expected[region]) > Decimal("0.01")}

    return not off, f"off by {off}" if off else "all regions within a cent"


TASK = Task("t08_quarter_revenue", "data", PROMPT, build, solve, check, short=True)
