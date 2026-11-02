# billing

Invoice arithmetic for the shop.

```python
from decimal import Decimal
from billing import Line, calc_total

calc_total([Line("mug", 2, Decimal("8.50"))], tax_rate=Decimal("0.20"))  # Decimal('20.40')
```

`calc_total` applies the discount before tax. `monthly_summary` adds up `calc_total` over a month's invoices.

From the shell: `python -m billing.cli 2:8.50 1:3.00`
