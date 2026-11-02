import unittest

from stockroom import Inventory, UnknownSku


class InventoryTests(unittest.TestCase):
    def test_add_and_quantity(self):
        inventory = Inventory()
        inventory.add("A-1", 5)
        inventory.add("A-1", 3)

        self.assertEqual(inventory.quantity("A-1"), 8)

    def test_unknown_sku(self):
        with self.assertRaises(UnknownSku):
            Inventory().quantity("nope")

    def test_sale_reduces_stock(self):
        inventory = Inventory()
        inventory.add("A-1", 5)
        inventory.record_sale("A-1", 2, day=10)

        self.assertEqual(inventory.quantity("A-1"), 3)
        self.assertEqual(inventory.sales("A-1"), [(2, 10)])

    def test_cannot_oversell(self):
        inventory = Inventory()
        inventory.add("A-1", 1)

        with self.assertRaises(ValueError):
            inventory.record_sale("A-1", 2, day=1)


if __name__ == "__main__":
    unittest.main()
