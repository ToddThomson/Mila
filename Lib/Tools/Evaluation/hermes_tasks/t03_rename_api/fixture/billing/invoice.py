from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal


@dataclass(frozen=True)
class Line:
    description: str
    quantity: int
    unit_price: Decimal


def calc_total(lines, tax_rate, discount=Decimal("0")):
    """Invoice total: the lines' sum, less `discount` (a fraction), plus tax on what remains, rounded to cents."""
    subtotal = sum((line.unit_price * line.quantity for line in lines), Decimal("0"))
    discounted = subtotal * (1 - discount)
    total = discounted * (1 + tax_rate)

    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
