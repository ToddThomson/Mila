import billing.invoice as invoice


def to_csv_row(number, customer, lines, tax_rate):
    """One invoice as a row for the accountant's spreadsheet."""
    return f"{number},{customer},{len(lines)},{invoice.calc_total(lines, tax_rate)}"
