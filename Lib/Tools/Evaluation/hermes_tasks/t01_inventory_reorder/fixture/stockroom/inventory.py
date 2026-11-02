"""Stock levels and sales for a small shop. Days are integers counted from the shop's opening day."""


class UnknownSku(KeyError):
    pass


class Inventory:
    def __init__(self):
        self._quantity = {}
        self._sales = []  # (sku, quantity, day)

    def add(self, sku, quantity):
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        self._quantity[sku] = self._quantity.get(sku, 0) + quantity

    def quantity(self, sku):
        if sku not in self._quantity:
            raise UnknownSku(sku)

        return self._quantity[sku]

    def skus(self):
        return sorted(self._quantity)

    def record_sale(self, sku, quantity, day):
        """Take `quantity` of `sku` out of stock, sold on `day`."""
        if sku not in self._quantity:
            raise UnknownSku(sku)

        if quantity <= 0:
            raise ValueError("quantity must be positive")

        if quantity > self._quantity[sku]:
            raise ValueError(f"only {self._quantity[sku]} of {sku} in stock")

        self._quantity[sku] -= quantity
        self._sales.append((sku, quantity, day))

    def sales(self, sku):
        return [(quantity, day) for sold, quantity, day in self._sales if sold == sku]
