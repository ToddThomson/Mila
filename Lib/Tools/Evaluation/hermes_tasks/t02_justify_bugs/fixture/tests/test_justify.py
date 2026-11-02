import unittest

from layout import justify, lines


class JustifyTests(unittest.TestCase):
    def test_lines_fill_greedily(self):
        self.assertEqual(lines("aa bb cc dd", 5), [["aa", "bb"], ["cc", "dd"]])

    def test_last_line_is_not_padded(self):
        self.assertEqual(justify("aa bb cc", 5), "aa bb\ncc")

    def test_extra_spaces_go_to_the_left_gaps(self):
        self.assertEqual(
            justify("The quick brown fox jumps over the lazy dog", 16),
            "The  quick brown\nfox  jumps  over\nthe lazy dog")

    def test_word_longer_than_width_is_refused(self):
        with self.assertRaises(ValueError):
            justify("antidisestablishment", 10)


if __name__ == "__main__":
    unittest.main()
