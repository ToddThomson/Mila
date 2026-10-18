# Reorder list

We keep running out of the fast sellers. Please add `Inventory.reorder_list(today, threshold_days)` so the
morning report can show what to reorder.

- Average daily sales for a SKU is everything sold in the 28 days ending on `today` (so days `today - 27`
  through `today`, inclusive), divided by 28. Sales after `today` don't count.
- "Days of cover" is the quantity in stock divided by that average.
- Return the SKUs whose days of cover is below `threshold_days`, most urgent first (fewest days of cover). Two
  SKUs with the same cover come in SKU order.
- A SKU that sold nothing in the window never needs reordering, even when it's out of stock.
- A SKU that is out of stock but did sell in the window has zero days of cover and is the most urgent.

Please add tests for it too.
