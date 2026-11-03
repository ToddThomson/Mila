#!/usr/bin/env python3
"""
Which folders of Lib/Src may import which.

A folder is a layer, and every module belongs to the folder its file lies in, since module
names do not follow the tree. One build target holds them all, so nothing but this check
stops a layer importing one above it.

ALLOWED is the graph as it stands, not an ideal: Dnn, Data and Deployment import each other.
What it holds firm is that nothing imports Agents, and nothing grows that was not there.

    python Publishing/hooks/check_layers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SOURCE = Path("Lib") / "Src"

# The folder a module's file lies in -> the folders it may import from. Its own is always allowed.
ALLOWED = {
    "Logging": set(),
    "Utils": set(),
    "Distribution": {"Utils", "(root)"},
    "Data": {"Dnn", "Logging", "Utils"},
    "Dnn": {"Data", "Deployment", "Logging", "Utils"},
    "Deployment": {"Data", "Distribution", "Dnn"},
    "Agents": {"Data", "Deployment", "Distribution", "Dnn", "Utils"},
}

MODULE = re.compile(r"^\s*export\s+module\s+([\w.]+)", re.MULTILINE)
IMPORT = re.compile(r"^\s*(?:export\s+)?import\s+([\w.]+)\s*;", re.MULTILINE)


def layer_of(path: Path, source: Path) -> str:
    parts = path.relative_to(source).parts

    return parts[0] if len(parts) > 1 else "(root)"


def check(root: Path) -> list[str]:
    source = root / SOURCE
    files = [(path, path.read_text(encoding="utf-8", errors="replace")) for path in sorted(source.rglob("*.ixx"))]
    owner = {}
    problems = []

    for path, text in files:
        match = MODULE.search(text)

        if match:
            owner[match.group(1)] = layer_of(path, source)

    for path, text in files:
        layer = layer_of(path, source)

        # The umbrella, Mila.ixx, re-exports every layer.
        if layer == "(root)":
            continue

        if layer not in ALLOWED:
            problems.append(f"{path.relative_to(root).as_posix()}: the folder {layer} has no entry in ALLOWED")
            continue

        for imported in IMPORT.findall(text):
            imported_layer = owner.get(imported)

            if imported_layer in (None, layer):
                continue

            if imported_layer not in ALLOWED[layer]:
                problems.append(f"{path.relative_to(root).as_posix()}: {layer} may not import {imported} ({imported_layer})")

    return problems


def main() -> int:
    problems = check(Path.cwd())

    for problem in problems:
        print(problem)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
