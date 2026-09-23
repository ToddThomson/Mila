#!/usr/bin/env python3
"""
The offline checks, run on exactly what is being committed.

Three checks that need no toolchain, no network and no GPU: NOTICE.md matches the
dependency pins, the sixteen release-version sites agree, and Doxygen builds without a
warning. Together they take about five seconds, so they run before every commit instead
of after a push.

They run against the STAGED tree, exported to a temporary directory, not the working
tree. A commit that stages half a change is judged on the half it carries.

    python scripts/hooks/commit_checks.py              run the checks on the staged tree
    python scripts/hooks/commit_checks.py --worktree   run them on the working tree instead
    python scripts/hooks/commit_checks.py --install    install the git pre-commit hook

A failing check blocks the commit. `git commit --no-verify` bypasses the hook when a
commit genuinely has to land broken.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HOOK_MARKER = "# Installed by scripts/hooks/commit_checks.py"

# Written into .git/hooks, which git does not track, so the hook itself stays a shim and
# every behaviour lives in this tracked file. `py` first: on Windows `python` can resolve to
# whatever virtual environment last put itself on PATH.
HOOK_SCRIPT = f"""#!/bin/sh
{HOOK_MARKER} -- edit that file, not this one.
root=$(git rev-parse --show-toplevel)
for python in "py -3" python3 python; do
    if command -v ${{python%% *}} >/dev/null 2>&1; then
        exec $python "$root/scripts/hooks/commit_checks.py"
    fi
done
echo "pre-commit: no Python found; the commit checks did not run." >&2
exit 1
"""


def export_staged_tree(destination: Path) -> None:
    prefix = destination.as_posix().rstrip("/") + "/"
    subprocess.run(
        ["git", "checkout-index", "--all", f"--prefix={prefix}"],
        cwd=REPO_ROOT,
        check=True,
    )


def pinned_doxygen_version(root: Path) -> str | None:
    workflow = root / ".github" / "workflows" / "publish-site.yml"
    match = re.search(r"^\s*DOXYGEN_VERSION:\s*(\S+)", workflow.read_text(encoding="utf-8"), re.MULTILINE)

    return match.group(1) if match else None


def run_check(name: str, command: list[str], root: Path, stdin: str | None = None) -> bool:
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started

    if result.returncode == 0:
        print(f"  pass  {name} ({elapsed:.1f}s)")
        return True

    print(f"  FAIL  {name} ({elapsed:.1f}s)")

    for line in (result.stdout + result.stderr).strip().splitlines():
        print(f"        {line}")

    return False


def run_doxygen(root: Path, output: Path) -> bool:
    doxygen = shutil.which("doxygen")

    if doxygen is None:
        print("  FAIL  Doxygen builds without warnings -- doxygen is not on PATH")
        return False

    installed = subprocess.run([doxygen, "--version"], capture_output=True, text=True).stdout.split()[0]
    pinned = pinned_doxygen_version(root)

    # A different Doxygen can warn where the publishing one does not, or stay quiet where it
    # warns, so a mismatch is reported but the check still runs.
    if pinned and installed != pinned:
        print(f"  note  doxygen {installed} is installed; the site publishes with {pinned}")

    # The Doxyfile's own OUTPUT_DIRECTORY is relative to the tree; a later assignment on
    # stdin wins, which keeps the generated HTML out of the checkout.
    configuration = (root / "Mila" / "Docs" / "Doxyfile").read_text(encoding="utf-8")
    configuration += f"\nOUTPUT_DIRECTORY = {output.as_posix()}\n"

    return run_check("Doxygen builds without warnings", [doxygen, "-"], root, stdin=configuration)


def run_checks(root: Path, scratch: Path) -> bool:
    python = sys.executable
    results = [
        run_check(
            "NOTICE.md matches the dependency pins",
            [python, "scripts/dependencies/check_pins.py", "--verify-notice"],
            root,
        ),
        run_check(
            "The release version is written consistently",
            [python, "scripts/release/version_sites.py", "--check"],
            root,
        ),
        run_doxygen(root, scratch / "docs"),
    ]

    return all(results)


def install_hook() -> int:
    hooks = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    if not hooks.is_absolute():
        hooks = REPO_ROOT / hooks

    hook = hooks / "pre-commit"

    # Git LFS installs post-checkout, post-commit, post-merge and pre-push, never pre-commit,
    # so the slot is normally free. Anything else already there is someone's, and is kept.
    if hook.exists() and HOOK_MARKER not in hook.read_text(encoding="utf-8", errors="replace"):
        print(f"{hook} already exists and was not installed by this script; leaving it alone.")
        return 1

    hook.write_text(HOOK_SCRIPT, encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    print(f"Installed {hook}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worktree", action="store_true", help="check the working tree instead of the staged tree")
    parser.add_argument("--install", action="store_true", help="install the git pre-commit hook")
    arguments = parser.parse_args()

    if arguments.install:
        return install_hook()

    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="mila-commit-checks-") as temporary:
        scratch = Path(temporary)

        if arguments.worktree:
            print("Commit checks on the working tree:")
            root = REPO_ROOT
        else:
            print("Commit checks on the staged tree:")
            root = scratch / "tree"
            root.mkdir()
            export_staged_tree(root)

        passed = run_checks(root, scratch)

    elapsed = time.monotonic() - started

    if passed:
        print(f"All commit checks passed ({elapsed:.1f}s).")
        return 0

    print(f"Commit blocked: a check failed ({elapsed:.1f}s). Fix it, or commit with --no-verify to bypass.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
