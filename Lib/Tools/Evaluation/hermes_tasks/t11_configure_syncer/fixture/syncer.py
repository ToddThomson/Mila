"""
syncer 2.1 -- copies new and changed files from a source directory into an archive, keeping older versions.

    python3 syncer.py            sync once, using syncer.ini
    python3 syncer.py --check    validate syncer.ini without copying anything
"""

import configparser
import fnmatch
import shutil
import sys
from pathlib import Path

RENAMED = {"src": "source", "dst": "destination", "ignore": "exclude", "versions": "keep_versions"}
KNOWN = {"source", "destination", "exclude", "keep_versions"}


def load(path=Path("syncer.ini")):
    if not path.is_file():
        raise SystemExit(f"error: {path} not found (see README.md)")

    parser = configparser.ConfigParser()
    parser.read(path)

    if "sync" not in parser:
        raise SystemExit("error: syncer.ini needs a [sync] section")

    section = parser["sync"]

    for key in section:
        if key in RENAMED:
            raise SystemExit(f"error: '{key}' was renamed to '{RENAMED[key]}' in syncer 2.0")

        if key not in KNOWN:
            raise SystemExit(f"error: unknown setting '{key}'")

    for key in ("source", "destination"):
        if key not in section:
            raise SystemExit(f"error: '{key}' is required")

    source = Path(section["source"])

    if not source.is_dir():
        raise SystemExit(f"error: source '{source}' is not a directory")

    try:
        keep = int(section.get("keep_versions", "1"))
    except ValueError:
        raise SystemExit("error: keep_versions must be a whole number")

    if keep < 1:
        raise SystemExit("error: keep_versions must be at least 1")

    exclude = [pattern.strip() for pattern in section.get("exclude", "").split(",") if pattern.strip()]

    return source, Path(section["destination"]), exclude, keep


def sync(source, destination, exclude, keep):
    destination.mkdir(parents=True, exist_ok=True)

    for path in sorted(source.iterdir()):
        if not path.is_file() or any(fnmatch.fnmatch(path.name, pattern) for pattern in exclude):
            continue

        target = destination / path.name

        if target.exists() and target.read_bytes() == path.read_bytes():
            continue

        if target.exists():
            for version in range(keep - 1, 0, -1):
                older = destination / f"{path.name}.{version}"

                if older.exists():
                    older.replace(destination / f"{path.name}.{version + 1}")

            target.replace(destination / f"{path.name}.1")

            for stale in destination.glob(f"{path.name}.*"):
                suffix = stale.name[len(path.name) + 1:]

                if suffix.isdigit() and int(suffix) >= keep:
                    stale.unlink()

        shutil.copy2(path, target)
        print(f"copied {path.name}")


def main(arguments):
    source, destination, exclude, keep = load()

    if arguments == ["--check"]:
        print(f"config OK: {source} -> {destination}, excluding {exclude or 'nothing'}, keeping {keep}")
        return 0

    sync(source, destination, exclude, keep)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
