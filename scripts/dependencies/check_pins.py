#!/usr/bin/env python3
"""Keep the dependency pins and NOTICE.md from drifting apart, and say when a pin is stale.

Two checks, deliberately different in kind:

  --verify-notice   Does NOTICE.md describe what the build actually fetches?
                    Deterministic, offline, and a HARD failure. NOTICE.md is a release
                    artifact -- Mila publishes wheels and container images that link
                    these libraries -- so a pin bump without the matching NOTICE edit
                    ships a false statement about what the binary contains. This check
                    exists because that drift has already happened twice.

  --check-upstream  Is any pin behind its latest upstream release? Needs network, and is
                    ADVISORY only. CI has no GPU (see build-pipeline.yml), so a CUTLASS
                    bump cannot be validated automatically; the honest output is a report
                    a human acts on, not a green pull request.

Run the first one before committing:

    python scripts/dependencies/check_pins.py --verify-notice
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_CMAKE = os.path.join(REPO_ROOT, "CMakeLists.txt")
TESTS_CMAKE = os.path.join(REPO_ROOT, "Mila", "Tests", "CMakeLists.txt")
NOTICE = os.path.join(REPO_ROOT, "NOTICE.md")

GITHUB_SLUG = re.compile(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def strip_comments(text):
    """Drop whole-line CMake comments.

    The root CMakeLists keeps a commented-out sentencepiece block; parsing it would
    invent a dependency the build never fetches.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def find_cpm_blocks(text):
    """Yield the body of each CPMAddPackage( ... ) call, paren-balanced.

    A regex cannot be trusted here: the OPTIONS lists are free-form strings and a
    non-greedy match would stop at the first close paren inside one.
    """
    for match in re.finditer(r"CPMAddPackage\s*\(", text):
        depth = 1
        index = match.end()

        while index < len(text) and depth > 0:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1

        yield text[match.end(): index - 1]


def parse_cmake_pins():
    """Every dependency the build fetches, keyed by its owner/repo slug."""
    pins = {}

    with open(ROOT_CMAKE, encoding="utf-8") as handle:
        root = strip_comments(handle.read())

    for body in find_cpm_blocks(root):
        repo = re.search(r"GITHUB_REPOSITORY\s+(\S+)", body)

        if not repo:
            continue

        # GIT_TAG wins when both are present: it is what CPM actually resolves.
        tag = re.search(r"GIT_TAG\s+(\S+)", body)
        version = re.search(r"^\s*VERSION\s+(\S+)", body, re.M)
        pin = tag.group(1) if tag else (version.group(1) if version else None)

        if pin:
            pins[repo.group(1)] = pin

    # googletest arrives through FetchContent with a tag embedded in an archive URL,
    # not through CPM, so it needs its own read or it silently escapes both checks.
    if os.path.exists(TESTS_CMAKE):
        with open(TESTS_CMAKE, encoding="utf-8") as handle:
            tests = strip_comments(handle.read())

        gtest = re.search(
            r"github\.com/(google/googletest)/archive/refs/tags/([^/\s\"]+)\.zip", tests)

        if gtest:
            pins[gtest.group(1)] = gtest.group(2)

    return pins


def parse_notice_rows():
    """The dependency table in NOTICE.md, keyed by owner/repo slug."""
    rows = {}

    with open(NOTICE, encoding="utf-8") as handle:
        for line in handle:
            if not line.lstrip().startswith("|"):
                continue

            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]

            if len(cells) < 2:
                continue

            slug = GITHUB_SLUG.search(cells[0])

            if not slug:
                continue

            # The version cell is written as markdown code for a branch name.
            rows[slug.group(1)] = cells[1].strip("`").strip()

    return rows


def verify_notice():
    pins = parse_cmake_pins()
    rows = parse_notice_rows()

    problems = []

    for slug, pin in sorted(pins.items()):
        if slug not in rows:
            problems.append(
                f"{slug}: fetched at {pin}, but NOTICE.md has no row for it")
        elif rows[slug] != pin:
            problems.append(
                f"{slug}: build fetches {pin}, NOTICE.md says {rows[slug]}")

    # A row for something no longer fetched is also drift, and the direction that
    # quietly overstates what a binary contains.
    for slug in sorted(rows):
        if slug not in pins:
            problems.append(
                f"{slug}: NOTICE.md lists it, but no CMake pin fetches it")

    print(f"Checked {len(pins)} fetched dependencies against NOTICE.md.\n")

    for slug, pin in sorted(pins.items()):
        state = "ok" if rows.get(slug) == pin else "DRIFT"
        print(f"  {state:<6} {slug:<26} {pin}")

    if problems:
        print("\nNOTICE.md does not match the build:\n")

        for problem in problems:
            print(f"  - {problem}")

        print("\nNOTICE.md records what Mila's published wheels and container images")
        print("link. Update it in the same commit as the pin.")

        return 1

    print("\nNOTICE.md matches the build.")

    return 0


def latest_release(slug, token):
    """Latest non-prerelease tag for a repo, or None if the API will not say."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{slug}/releases?per_page=20",
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "mila-check-pins"})

    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            releases = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as error:
        print(f"  ?      {slug:<26} could not query upstream: {error}")
        return None

    for release in releases:
        # Skip drafts, prereleases, and the "dev" tags NVIDIA publishes between
        # stable CUTLASS releases -- recommending one of those would be wrong.
        if release.get("draft") or release.get("prerelease"):
            continue

        tag = release.get("tag_name", "")

        if tag and "dev" not in tag.lower():
            return tag

    return None


def check_upstream():
    pins = parse_cmake_pins()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    behind = []

    print(f"Comparing {len(pins)} pins against upstream.\n")

    for slug, pin in sorted(pins.items()):
        latest = latest_release(slug, token)

        if latest is None:
            continue

        # CPM's VERSION form drops the tag's leading "v" (nlohmann_json VERSION 3.12.0
        # resolves the v3.12.0 tag), so compare without it or every such pin reads BEHIND
        # against itself.
        if latest.lstrip("v") == pin.lstrip("v"):
            print(f"  ok     {slug:<26} {pin}")
        else:
            print(f"  BEHIND {slug:<26} {pin}  ->  {latest}")
            behind.append((slug, pin, latest))

    # Not a moving branch, but worth naming: a branch pin makes no tagged release
    # reproducible, which no version comparison would surface.
    floating = [(slug, pin) for slug, pin in sorted(pins.items())
                if pin in ("master", "main", "HEAD")]

    if floating:
        print("\nPinned to a moving branch (no tagged build is reproducible):\n")

        for slug, pin in floating:
            print(f"  - {slug} is pinned to {pin}")

    if behind:
        print(f"\n{len(behind)} pin(s) behind upstream. Advisory only -- a bump that")
        print("touches CUDA cannot be validated by CI, which has no GPU.")

    # Advisory: never fail a build on this.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-notice", action="store_true",
                        help="fail if NOTICE.md disagrees with the CMake pins (offline)")
    parser.add_argument("--check-upstream", action="store_true",
                        help="report pins behind their latest upstream release (network)")

    args = parser.parse_args()

    if not args.verify_notice and not args.check_upstream:
        parser.print_help()
        return 2

    status = 0

    if args.verify_notice:
        status |= verify_notice()

    if args.check_upstream:
        if args.verify_notice:
            print()

        status |= check_upstream()

    return status


if __name__ == "__main__":
    sys.exit(main())
