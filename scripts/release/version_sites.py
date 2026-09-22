#!/usr/bin/env python3
"""
The release version, everywhere it is written by hand.

Sixteen sites across eight files name a published release: a FetchContent GIT_TAG a
reader copies, an image tag a reader runs, sample output, a status line. None of them
derive from Version.txt, and nothing checked them until this script existed -- they
went stale twice, once pointing a downstream consumer at an unreleased tag, once
leaving the website's C++ tab pinned a checkpoint behind the output printed beside it.

WHAT THIS DOES NOT TOUCH, and why a find-replace is the wrong tool. The tree also
carries historical references to the same strings -- "removed in 0.20.0-beta.3",
"Resolved at 0.20.0-rc.1+15", provenance lines in the specifications. Those are
statements about the past and must not move. Every site below is therefore anchored to
its own surrounding text rather than matched by version pattern alone.

THE SITES NAME THE LAST PUBLISHED RELEASE, NOT THE WORKING VERSION. Mid-cycle, dev
carries 0.21.0-dev+26 while these correctly still point at v0.20.0, because that is the
tag a reader can actually fetch. So the always-true invariant is that the sixteen agree
WITH EACH OTHER; equality with Version.txt is true only just after a release-prep
commit. Hence two checks, not one:

    --check                 the sixteen agree with each other         (CI, every commit)
    --check --expect X.Y.Z  ...and name exactly that release          (release step 2)
    --set X.Y.Z             rewrite all sixteen                       (release step 2)

A dev snapshot is never a release, so --expect reported against one has nothing to
compare and says so instead of failing.

Positioning prose is NOT here and cannot be: "Mila is in public beta" is a sentence a
production release makes false, and rewriting it is a judgement about what to say
instead. `--audit-prose` lists those sites so they are at least never forgotten.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# A version as it appears in a semver tag, without the leading v and without +build. The
# alpha/beta/rc alternation is kept although the ladder was retired at 0.21.0: a site names
# the last PUBLISHED release, and v0.20.0-beta.3 is one of those forever.
VERSION_PATTERN = r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?"

# What Version.txt may say, which is a superset: it names the version being BUILT, and
# mid-cycle that is a dev snapshot. Deliberately NOT folded into VERSION_PATTERN -- widening
# the site pattern would let --set write an unreleasable version into a FetchContent GIT_TAG
# and nothing downstream would object.
WORKING_VERSION_PATTERN = rf"(?:{VERSION_PATTERN}|\d+\.\d+\.\d+-dev)"


@dataclass(frozen=True)
class Site:
    """
    One hand-written occurrence of the release version.

    `pattern` must match exactly once in `path` and must capture the bare version --
    no leading v -- in group 'ver'. Anchoring is the whole point: the pattern carries
    enough surrounding text that it cannot also match a historical reference.
    """

    path: str
    pattern: str
    note: str


SITES: tuple[Site, ...] = (
    Site(
        "RELEASING.md",
        rf"Last checkpoint tagged: \*\*`v(?P<ver>{VERSION_PATTERN})`\*\*",
        "the procedure's own record of what shipped last",
    ),
    Site(
        "README.md",
        rf"Current release: `(?P<ver>{VERSION_PATTERN})`",
        "status callout on the page a visitor lands on",
    ),
    Site(
        "getting-started.md",
        rf"GIT_TAG\s+v(?P<ver>{VERSION_PATTERN})\s+# pin to a published release tag",
        "section 7 FetchContent pin, copy-pasted by readers",
    ),
    Site(
        "Mila/Samples/QuickStart/Cpp/CMakeLists.txt",
        rf"GIT_TAG\s+v(?P<ver>{VERSION_PATTERN}) # pin to a published release tag",
        "the published QuickStart the website links to",
    ),
    Site(
        "Mila/Samples/QuickStart/Cpp/README.md",
        rf"^Mila (?P<ver>{VERSION_PATTERN})\r?$",
        "sample output a reader compares against",
    ),
    Site(
        "Mila/Samples/QuickStart/Cpp/README.md",
        rf"GIT_TAG\s+v(?P<ver>{VERSION_PATTERN})\s+# pin to a published release tag",
        "FetchContent pin",
    ),
    Site(
        "Mila/Samples/QuickStart/Cpp/README.md",
        rf"archive/refs/tags/v(?P<ver>{VERSION_PATTERN})\.zip",
        "release archive URL",
    ),
    Site(
        "Web/layouts/index.html",
        rf"GIT_TAG\s+v(?P<ver>{VERSION_PATTERN})",
        "C++ tab FetchContent pin",
    ),
    Site(
        "Web/layouts/index.html",
        rf"<code>Mila (?P<ver>{VERSION_PATTERN})",
        "C++ tab sample output -- went out of step with the pin above it",
    ),
    Site(
        "Web/layouts/index.html",
        rf"mila-llm:(?P<ver>{VERSION_PATTERN})-devel",
        "Docker tab image tag",
    ),
    Site(
        "Web/layouts/index.html",
        rf"mila-llm:(?P<ver>{VERSION_PATTERN})-runtime install",
        "evaluate band, install command",
    ),
    Site(
        "Web/layouts/index.html",
        rf"mila-llm:(?P<ver>{VERSION_PATTERN})-runtime chat",
        "evaluate band, chat command",
    ),
    Site(
        "scripts/dockerhub/verify-image.sh",
        rf'MILA_IMAGE:=mila-llm:(?P<ver>{VERSION_PATTERN})-runtime',
        "default image, or the script verifies the previous release",
    ),
    Site(
        "scripts/dockerhub/overview.md",
        rf"mila-llm:(?P<ver>{VERSION_PATTERN})-runtime install",
        "Docker Hub overview, install command",
    ),
    Site(
        "scripts/dockerhub/overview.md",
        rf"mila-llm:(?P<ver>{VERSION_PATTERN})-runtime chat",
        "Docker Hub overview, chat command",
    ),
    Site(
        "scripts/dockerhub/overview.md",
        rf"mila-llm:(?P<ver>{VERSION_PATTERN})-devel",
        "Docker Hub overview, devel image",
    ),
)

# Sentences a production release makes false. Not rewritable by pattern -- what to say
# instead is a judgement -- so they are listed, never edited, and reported by --audit-prose.
PROSE_SITES: tuple[tuple[str, str], ...] = (
    ("README.md", "Current release:"),
    ("README.md", "Current Status"),
    ("README.md", "first production release"),
    ("CLAUDE.md", "first production release"),
    # Falsified at 1.0 rather than at the next release, but the audit is the only
    # place that remembers they exist.
    ("README.md", "Pre-1.0"),
    ("getting-started.md", "Pre-1.0"),
    ("SECURITY.md", "pre-1.0"),
)


def release_version_from_file() -> str:
    """Version.txt with any +build metadata dropped -- a tag never carries it."""
    raw = (REPO_ROOT / "Version.txt").read_text(encoding="utf-8").strip()

    return raw.split("+", 1)[0]


def validate_version(version: str, *, allow_dev: bool = False) -> str:
    """
    `allow_dev` is the difference between the two callers. --set writes a release version
    into sites a reader copies, so a dev snapshot there is a defect; --expect only reads
    Version.txt, where a dev snapshot is the normal mid-cycle state.
    """
    pattern = WORKING_VERSION_PATTERN if allow_dev else VERSION_PATTERN

    if not re.fullmatch(pattern, version):
        expected = "X.Y.Z or X.Y.Z-dev" if allow_dev else "X.Y.Z"

        raise SystemExit(
            f"'{version}' is not a release version. Expected {expected}, with no +build "
            "metadata -- a tag never carries it."
        )

    return version


def read_exact(path: Path) -> str:
    """Read without newline translation, so a rewrite cannot silently convert CRLF to LF."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_exact(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def find_all() -> list[tuple[Site, str]]:
    """Every site paired with the version it currently names. Aborts on a site that moved."""
    found: list[tuple[Site, str]] = []

    for site in SITES:
        path = REPO_ROOT / site.path
        text = read_exact(path)
        matches = list(re.finditer(site.pattern, text, re.MULTILINE))

        if len(matches) != 1:
            raise SystemExit(
                f"{site.path}: expected exactly one match for the {site.note!r} site, "
                f"found {len(matches)}.\n"
                "  The file changed shape. Fix the pattern in scripts/release/version_sites.py "
                "rather than editing the site by hand -- an unmatched site is one nothing checks."
            )

        found.append((site, matches[0].group("ver")))

    return found


def command_check(expect: str | None) -> int:
    found = find_all()
    versions = {version for _, version in found}

    if len(versions) > 1:
        print("FAIL: the release version is not written consistently.\n")

        for site, version in sorted(found, key=lambda pair: pair[1]):
            print(f"  {version:<20} {site.path}  ({site.note})")

        print("\nA reader copies one of these and gets a different release from the one beside it.")

        return 1

    current = versions.pop()

    # Mid-cycle Version.txt names a dev snapshot, and a snapshot is not something a site can
    # name -- so there is nothing to compare and the mismatch is the correct state. Reporting
    # it as a pass with a reason beats the old behaviour, where the argument validator rejected
    # `0.21.0-dev` outright and read like the tree was broken.
    if expect is not None and expect.endswith("-dev"):
        print(f"OK: {len(found)} sites, all naming {current}.")
        print(
            f"\n{expect} is a dev snapshot, so --expect has nothing to compare. The sites name\n"
            "the last release a reader can fetch; they move at release step 2, once Version.txt\n"
            "has been set to the release version."
        )

        return 0

    if expect is not None and current != expect:
        print(
            f"FAIL: the {len(found)} sites consistently name {current}, but this release is "
            f"{expect}.\n\n  Run:  python scripts/release/version_sites.py --set {expect}"
        )

        return 1

    print(f"OK: {len(found)} sites, all naming {current}.")

    return 0


def command_set(version: str) -> int:
    edits: dict[Path, str] = {}
    changed = 0

    for site in SITES:
        path = REPO_ROOT / site.path
        text = edits.get(path, read_exact(path))

        def replace(match: re.Match[str]) -> str:
            span_start, span_end = match.span("ver")

            return match.group(0)[: span_start - match.start()] + version + match.group(0)[span_end - match.start():]

        new_text, count = re.subn(site.pattern, replace, text, flags=re.MULTILINE)

        if count != 1:
            raise SystemExit(f"{site.path}: {count} matches for the {site.note!r} site, expected 1.")

        if new_text != text:
            changed += 1

        edits[path] = new_text

    for path, text in edits.items():
        write_exact(path, text)

    print(f"Set {len(SITES)} sites to {version} across {len(edits)} files ({changed} changed).")
    print("\nNOT done by this script, and owed by a production release:")
    print("  - Version.txt itself (the release-prep commit sets it)")
    print("  - the positioning prose: python scripts/release/version_sites.py --audit-prose")

    return 0


def command_audit_prose() -> int:
    print("Sentences a release may make false. This script never edits them.\n")

    # Several needles legitimately hit the same sentence, so report each line once --
    # this output is a checklist somebody works through by hand.
    seen: set[tuple[str, int]] = set()

    for path_name, needle in PROSE_SITES:
        path = REPO_ROOT / path_name

        if not path.is_file():
            print(f"  {path_name}: MISSING")
            continue

        for number, line in enumerate(read_exact(path).splitlines(), start=1):
            if needle in line and (path_name, number) not in seen:
                seen.add((path_name, number))
                print(f"  {path_name}:{number}  {line.strip()[:96]}")

    return 0


def main() -> int:
    # The sites quote prose carrying em dashes; a Windows console defaults to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="assert the sites agree with each other")
    group.add_argument("--set", metavar="VERSION", help="rewrite every site to this release version")
    group.add_argument("--audit-prose", action="store_true", help="list positioning prose a release may falsify")
    parser.add_argument(
        "--expect",
        metavar="VERSION",
        nargs="?",
        const="",
        help="with --check: also assert the sites name this release (no value: read Version.txt)",
    )
    arguments = parser.parse_args()

    if arguments.set:
        return command_set(validate_version(arguments.set))

    if arguments.audit_prose:
        return command_audit_prose()

    expect = arguments.expect

    if expect == "":
        expect = release_version_from_file()

    return command_check(validate_version(expect, allow_dev=True) if expect else None)


if __name__ == "__main__":
    sys.exit(main())
