"""Publish a Mila model card and its weights to the HuggingFace Hub.

Every failure during the first manual publish was a mismatch nobody checked: a stale
digest, a path that differed between the manifest and the upload, a file quietly missing.
So this validates before it uploads and verifies after, and it is safe to re-run --
anything already correct on the Hub is skipped.

Usage:
    python publish_model.py <card-directory> [--dry-run]

The card directory holds the files published verbatim (README.md, mila.json, LICENSE)
plus publish.json, which names the repository and maps Hub paths to local sources for
the large files that live outside the repository.

Authentication: HF_TOKEN, MILA_HF_TOKEN, or a stored `hf auth login` token. The token
needs write access to the target repository and is never printed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    sys.exit(
        "huggingface_hub is not installed.\n"
        "Install it into the converter venv:\n"
        "  Mila/Tools/Converters/.venv/Scripts/python.exe -m pip install -U huggingface_hub"
    )

# Files in the card directory that are published verbatim, in upload order. Small, so
# they go first: a mistake in them is then visible before a multi-gigabyte transfer.
CARD_FILES = ["mila.json", "README.md", "LICENSE"]

# publish.json describes the upload; it is not itself published.
EXCLUDED_FROM_CARD = {"publish.json"}


def sha256_of(path: Path, chunk=1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(card_dir: Path) -> dict:
    manifest_path = card_dir / "mila.json"
    if not manifest_path.is_file():
        sys.exit(f"No mila.json in {card_dir}. Run ExportArtifact --emit-manifest first.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def declared_files(manifest: dict) -> dict:
    """Hub path -> (sha256, bytes) for every file any variant declares."""
    out = {}
    for variant in manifest.get("variants", {}).values():
        for entry in variant.get("files", {}).values():
            out[entry["path"]] = (entry.get("sha256"), entry.get("bytes"))
    return out


def validate(card_dir: Path, repo_root: Path, manifest: dict, sources: dict) -> list:
    """Check every declared file exists locally and matches its recorded digest.

    A manifest that disagrees with the bytes produces a repository that fails
    verification on every download, so this is a hard gate rather than a warning.
    """
    problems = []
    resolved = []

    for hub_path, (expected_sha, expected_bytes) in declared_files(manifest).items():
        local = card_dir / hub_path
        if not local.is_file():
            mapped = sources.get(hub_path)
            if mapped is None:
                problems.append(
                    f"{hub_path}: declared in mila.json but not in the card directory "
                    f"and not mapped in publish.json"
                )
                continue
            local = (repo_root / mapped).resolve()
            if not local.is_file():
                problems.append(f"{hub_path}: mapped to {local}, which does not exist")
                continue

        actual_bytes = local.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            problems.append(
                f"{hub_path}: manifest says {expected_bytes} bytes, file is {actual_bytes}. "
                f"Re-run ExportArtifact --emit-manifest."
            )
            continue

        print(f"  hashing {hub_path} ({actual_bytes / 1024**3:.2f} GB)...", flush=True)
        actual_sha = sha256_of(local)
        if expected_sha and actual_sha != expected_sha:
            problems.append(
                f"{hub_path}: manifest sha256 {expected_sha[:16]}... but file is "
                f"{actual_sha[:16]}.... Re-run ExportArtifact --emit-manifest."
            )
            continue

        print(f"    ok  sha256 {actual_sha[:16]}...")
        resolved.append((hub_path, local))

    return problems, resolved


def already_current(api: HfApi, repo_id: str, hub_path: str, local: Path) -> bool:
    """True when the Hub already holds this exact content, so the upload can be skipped."""
    try:
        info = api.get_paths_info(repo_id=repo_id, paths=[hub_path], repo_type="model")
    except HfHubHTTPError:
        return False

    if not info:
        return False

    entry = info[0]
    remote_sha = getattr(getattr(entry, "lfs", None), "sha256", None)
    if remote_sha is None:
        return False

    return remote_sha == sha256_of(local)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path,
                        help="Model card directory, or a package directory built by "
                             "ExportArtifact --package")
    parser.add_argument("--repo", help="Target repository as <owner>/<name>. Required for a "
                                       "package directory, which carries no publish.json.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report, upload nothing")
    args = parser.parse_args()

    card_dir = args.card.resolve()
    if not card_dir.is_dir():
        sys.exit(f"Not a directory: {card_dir}")

    publish_path = card_dir / "publish.json"
    publish = {}

    if publish_path.is_file():
        publish = json.loads(publish_path.read_text(encoding="utf-8"))
    elif not args.repo:
        sys.exit(
            f"No publish.json in {card_dir}, and no --repo given.\n"
            f"A package directory holds every declared file, so it needs only the "
            f"repository: --repo <owner>/<name>."
        )

    repo_id = args.repo or publish["repo_id"]

    # Only mapped sources are resolved against it; a package directory has none, because it
    # holds the files it declares.
    repo_root = card_dir.parents[4] if len(card_dir.parents) > 4 else card_dir

    sources = {k: v for k, v in publish.get("sources", {}).items() if not k.startswith("//")}

    manifest = load_manifest(card_dir)

    print(f"Publishing {card_dir.name} -> {repo_id}")
    print("\nValidating declared files against mila.json")
    problems, resolved = validate(card_dir, repo_root, manifest, sources)

    if problems:
        print("\nValidation failed:", file=sys.stderr)
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        return 2

    card_uploads = [
        (name, card_dir / name)
        for name in CARD_FILES
        if (card_dir / name).is_file() and name not in EXCLUDED_FROM_CARD
    ]

    missing_card = [n for n in CARD_FILES if not (card_dir / n).is_file()]
    for name in missing_card:
        print(f"  note: {name} absent from the card directory, skipping")

    if args.dry_run:
        print("\nDry run. Would upload:")
        for hub_path, local in card_uploads + resolved:
            print(f"  {hub_path:<40} {local}")
        return 0

    api = HfApi()

    try:
        whoami = api.whoami()
    except Exception as error:
        print(f"\nNot authenticated: {error}", file=sys.stderr)
        print("Set HF_TOKEN, or run: hf auth login", file=sys.stderr)
        return 3

    print(f"\nAuthenticated as {whoami.get('name', '<unknown>')}")

    # Card files first: small, and a mistake in them is visible before a long transfer.
    for hub_path, local in card_uploads + resolved:
        if already_current(api, repo_id, hub_path, local):
            print(f"  skip    {hub_path} (already current)")
            continue

        size_gb = local.stat().st_size / 1024**3
        print(f"  upload  {hub_path} ({size_gb:.2f} GB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=hub_path,
            repo_id=repo_id,
            repo_type=publish.get("repo_type", "model"),
            commit_message=f"Publish {hub_path}",
        )

    print("\nVerifying the published repository")
    remote = {s.rfilename for s in api.model_info(repo_id, files_metadata=False).siblings}

    expected = {hub_path for hub_path, _ in card_uploads + resolved}
    missing = sorted(expected - remote)

    if missing:
        print("  ! missing from the repository after upload:", file=sys.stderr)
        for name in missing:
            print(f"      {name}", file=sys.stderr)
        return 4

    print(f"  {len(expected)} file(s) present")
    print(f"\nhttps://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
