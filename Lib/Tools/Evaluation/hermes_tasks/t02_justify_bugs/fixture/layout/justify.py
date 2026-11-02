"""Full justification for the plain-text newsletter."""

from layout.words import split_words


def lines(text, width):
    """Greedy fill: each line takes as many words as fit in `width` with one space between them."""
    rows, current = [], []

    for word in split_words(text):
        if len(word) > width:
            raise ValueError(f"'{word}' is longer than the line width {width}")

        if current and len(" ".join(current + [word])) > width:
            rows.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        rows.append(current)

    return rows


def justify(text, width):
    """
    Every line but the last is padded to exactly `width` by widening the gaps between its words; when the spaces
    do not divide evenly, the leftmost gaps get one more. The last line, and any line holding a single word, is
    left-aligned and not padded.
    """
    rows = lines(text, width)
    justified = []

    for index, words in enumerate(rows):
        if index == len(rows) - 1 or len(words) == 1:
            justified.append(" ".join(words))
            continue

        gaps = len(words) - 1
        base, extra = divmod(width - sum(len(word) for word in words), gaps)
        line = ""

        for position, word in enumerate(words[:-1]):
            line += word + " " * (base + (1 if position >= gaps - extra else 0))

        justified.append(line + words[-1])

    return "\n".join(justified)
