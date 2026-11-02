"""Coding: two bugs from a user's report, in two files, one of them caught by no test."""

from pathlib import Path

from hermes_tasks.common import Task, copy_fixture, fixture_path, not_blocked, run_hidden_tests

PROMPT = "Fix the bugs described in BUG_REPORT.md."

HIDDEN_TESTS = '''
import unittest

from layout import justify, lines


class JustifyHiddenTests(unittest.TestCase):
    def test_left_gaps_get_the_extra_space(self):
        self.assertEqual(justify("a bb ccc dddd eeeee", 11), "a   bb  ccc\\ndddd eeeee")
        self.assertEqual(justify("w1 w2 w3 w4 w5", 12), "w1  w2 w3 w4\\nw5")

    def test_any_whitespace_separates_words(self):
        self.assertEqual(justify("one  two\\nthree four", 9), "one   two\\nthree\\nfour")
        self.assertEqual(lines("Thanks for reading.  See you\\nnext week", 40),
                         [["Thanks", "for", "reading.", "See", "you", "next", "week"]])

    def test_tabs_and_surrounding_whitespace(self):
        self.assertEqual(justify("  a\\tb  ", 5), "a b")

    def test_single_word_line_is_not_padded(self):
        self.assertEqual(justify("abcdefgh ij", 8), "abcdefgh\\nij")

    def test_long_word_still_refused(self):
        with self.assertRaises(ValueError):
            justify("ok antidisestablishment", 10)


if __name__ == "__main__":
    unittest.main()
'''


def build(workdir: Path) -> None:
    copy_fixture(__file__, workdir)


def solve(workdir: Path) -> str:
    words = workdir / "layout" / "words.py"
    words.write_text(words.read_text().replace('text.split(" ")', "text.split()"))

    justify = workdir / "layout" / "justify.py"
    justify.write_text(justify.read_text().replace("position >= gaps - extra", "position < extra"))

    return "Fixed both."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    original_tests = (fixture_path(__file__) / "tests" / "test_justify.py").read_text()

    return run_hidden_tests(workdir, {"tests/test_justify_hidden.py": HIDDEN_TESTS},
                            restore={"tests/test_justify.py": original_tests})


TASK = Task("t02_justify_bugs", "coding", PROMPT, build, solve, check)
