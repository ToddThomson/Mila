#!/usr/bin/env python3
"""
Cut a Mila release, locally.

    python scripts/release/release.py prepare minor     set the release version on dev, uncommitted
    python scripts/release/release.py prepare patch
    python scripts/release/release.py finish            commit, build, ask, merge, tag, reopen dev

    python scripts/release/release.py build                  build and verify HEAD's release files
    python scripts/release/release.py build --commit <ref>   any commit
    python scripts/release/release.py build --resume         rerun only what has not passed
    python scripts/release/release.py build --only images    one stage (comma-separate several)

`prepare` edits the working tree and stops, because the prose a release makes false is a
judgement: review the diff, rewrite what it lists, then run `finish`. `finish` stops once
more, for a typed confirmation, before it changes master or creates a tag.

NOTHING IS PUSHED OR PUBLISHED. master, dev and the tag change in the local repository only,
and master is written without checking it out, so the working copy stays on dev throughout.

`build` works from a `git archive` export of the commit -- never the working tree -- in a
run directory outside the repository, and changes nothing in git. The four wheels and two
images are built once, and the files tested are the files that would ship. A stage's full
output goes to the run's logs/ directory; the console shows one line per stage, and every
run appends a line to history.log in the workspace.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

# The repository being released. The checkout running this script unless --repository names
# another -- which is how the git steps are exercised on a throwaway clone.
repository = REPO_ROOT

DEFAULT_WORKSPACE = REPO_ROOT.parent / "Mila-release"
WSL_DISTRIBUTION = os.environ.get("MILA_WSL_DISTRO", "Ubuntu-Dev")
WSL_WORKSPACE = "~/mila-release"

# Keep in step with requires-python in pyproject.toml and the interpreter lists in both wheel
# build scripts: one wheel per interpreter per platform.
INTERPRETERS = ("3.12", "3.13")

# Local, namespace-free image name: `docker push` of it cannot reach toddthomson/mila-llm.
LOCAL_IMAGE = "mila-release"

HEARTBEAT_SECONDS = 60

# The scripts that judge the built files, copied into each run from the checkout running this.
CHECKS = (
    Path("scripts/pypi/verify_wheel_cleanroom.py"),
    Path("scripts/dockerhub/verify-image.sh"),
    Path("scripts/dockerhub/verify-devel-image.sh"),
)


class StageFailed(Exception):
    pass


@dataclass
class Run:
    commit: str
    short: str
    version: str
    pep440: str
    directory: Path
    wsl_directory: str

    @property
    def source(self) -> Path:
        return self.directory / "src"

    @property
    def wheels(self) -> Path:
        return self.directory / "wheels"

    @property
    def logs(self) -> Path:
        return self.directory / "logs"

    @property
    def wsl_source(self) -> str:
        return f"{self.wsl_directory}/src"

    @property
    def checks(self) -> Path:
        return self.directory / "checks"

    @property
    def wsl_checks(self) -> str:
        return f"{self.wsl_directory}/checks"

    @property
    def verifier(self) -> Path:
        return self.checks / "verify_wheel_cleanroom.py"

    def image(self, target: str) -> str:
        return f"{LOCAL_IMAGE}:{self.short}-{target}"


# ---------------------------------------------------------------------------
# Running commands
# ---------------------------------------------------------------------------

def run_logged(command: list[str], log: Path, cwd: Path | None = None, environment: dict | None = None) -> None:
    """Run a command with its output in a log file, printing a heartbeat while it runs."""
    started = time.monotonic()
    next_heartbeat = started + HEARTBEAT_SECONDS

    with open(log, "a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT)

        while process.poll() is None:
            time.sleep(1)

            if time.monotonic() >= next_heartbeat:
                minutes = int((time.monotonic() - started) // 60)
                print(f"          ... still running ({minutes} min)", flush=True)
                next_heartbeat += HEARTBEAT_SECONDS

    if process.returncode != 0:
        raise StageFailed(f"exit {process.returncode}: {' '.join(command)}")


def wsl(script: str, log: Path) -> None:
    """Run a bash script in the WSL distribution, logged."""
    run_logged(["wsl", "-d", WSL_DISTRIBUTION, "--", "bash", "-lc", f"set -euo pipefail\n{script}"], log)


def wsl_capture(script: str) -> str:
    result = subprocess.run(
        ["wsl", "-d", WSL_DISTRIBUTION, "--", "bash", "-lc", script],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()

    return f"/mnt/{drive}" + resolved.as_posix()[len(resolved.drive):]


def git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
    )

    if check and result.returncode != 0:
        raise StageFailed(f"git {' '.join(arguments)} failed:\n{result.stderr.strip()}")

    return result.stdout.strip()


def git_succeeds(*arguments: str) -> bool:
    return subprocess.run(["git", *arguments], cwd=repository, capture_output=True).returncode == 0


def remove_tree(path: Path) -> None:
    # The CMake build trees nest deep enough to pass MAX_PATH; the extended-length prefix
    # lets rmtree reach them.
    if path.exists():
        shutil.rmtree("\\\\?\\" + str(path.resolve()) if os.name == "nt" else path)


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

def pep440_version(version: str) -> str:
    """The wheel's version, by the same rule as mila_pep440_version in cmake/MilaVersion.cmake."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-dev\+(\d+))?", version)

    if not match:
        raise StageFailed(f"Version.txt '{version}' is neither X.Y.Z nor X.Y.Z-dev+N")

    return match.group(1) if match.group(2) is None else f"{match.group(1)}.dev{match.group(2)}"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_export(run: Run, log: Path) -> None:
    remove_tree(run.source)
    run.source.mkdir(parents=True)
    archive = run.directory / "src.tar"

    subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), run.commit], cwd=repository, check=True)

    with tarfile.open(archive) as tar:
        tar.extractall(run.source, filter="data")

    # Linux builds run on the distribution's own filesystem; across /mnt they are several
    # times slower.
    wsl(
        f"rm -rf {run.wsl_source}\n"
        f"mkdir -p {run.wsl_source}\n"
        f"tar -xf '{windows_to_wsl(archive)}' -C {run.wsl_source}\n",
        log,
    )

    archive.unlink()


def copy_checks(run: Run, log: Path) -> None:
    # The checks are this tool's, not the release tree's: what is being judged is the built
    # files, and the judge is whatever version of the tooling is running. Copied on every
    # invocation, so a --resume after fixing a check uses the fixed one.
    remove_tree(run.checks)
    run.checks.mkdir()

    for check in CHECKS:
        shutil.copy2(REPO_ROOT / check, run.checks)

    wsl(f"rm -rf {run.wsl_checks}\nmkdir -p {run.wsl_checks}\ncp '{windows_to_wsl(run.checks)}'/* {run.wsl_checks}/", log)


def stage_windows_wheels(run: Run, log: Path) -> None:
    run_logged(
        ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(run.source / "scripts" / "pypi" / "build-wheel-windows.ps1")],
        log,
    )


def stage_linux_wheels(run: Run, log: Path) -> None:
    # Its own compose project, so the release build's volumes never mix with the ones an
    # everyday `docker compose` from the checkout uses.
    wsl(
        f"cd {run.wsl_source}/Docker\n"
        "docker compose -p mila-release -f docker-compose.wheel.yml build\n"
        "docker compose -p mila-release -f docker-compose.wheel.yml run --rm mila-wheel mila-build-wheel\n",
        log,
    )


def stage_collect_wheels(run: Run, log: Path) -> None:
    remove_tree(run.wheels)
    run.wheels.mkdir(parents=True)

    for wheel in (run.source / "out" / "wheel").glob("*.whl"):
        shutil.copy2(wheel, run.wheels)

    wsl(f"cp {run.wsl_source}/out/wheel/*.whl '{windows_to_wsl(run.wheels)}/'", log)

    found = sorted(wheel.name for wheel in run.wheels.glob("*.whl"))
    expected = []

    for interpreter in INTERPRETERS:
        tag = "cp" + interpreter.replace(".", "")
        expected.append(rf"mila_llm-{re.escape(run.pep440)}-{tag}-{tag}-win_amd64\.whl")
        expected.append(rf"mila_llm-{re.escape(run.pep440)}-{tag}-{tag}-manylinux_\d+_\d+_x86_64\.whl")

    with open(log, "a", encoding="utf-8") as handle:
        handle.write("\nwheels:\n" + "".join(f"  {name}\n" for name in found))

    unmatched = [pattern for pattern in expected if not any(re.fullmatch(pattern, name) for name in found)]

    if unmatched or len(found) != len(expected):
        raise StageFailed(
            f"expected exactly {len(expected)} wheels at {run.pep440}, found {len(found)}: {found}")


def wheel_for(run: Run, interpreter: str, platform: str) -> Path:
    tag = "cp" + interpreter.replace(".", "")
    matches = list(run.wheels.glob(f"mila_llm-{run.pep440}-{tag}-{tag}-{platform}*.whl"))

    if len(matches) != 1:
        raise StageFailed(f"no single {tag} {platform} wheel in {run.wheels}")

    return matches[0]


def stage_test_windows_wheels(run: Run, log: Path) -> None:
    # A machine with a CUDA Toolkit cannot prove the wheel works without one, so this asserts
    # the part it can: every CUDA library the process maps comes from the wheel's own
    # dependencies. CUDA_PATH and the Toolkit's PATH entries are removed as well, so nothing
    # but the package's own backstop can reach the host copy.
    environment = {key: value for key, value in os.environ.items() if key != "CUDA_PATH"}
    environment["PATH"] = os.pathsep.join(
        entry for entry in environment.get("PATH", "").split(os.pathsep)
        if "NVIDIA GPU Computing Toolkit" not in entry)

    for interpreter in INTERPRETERS:
        venv = run.directory / f"test-venv-{interpreter}"
        remove_tree(venv)
        python = venv / "Scripts" / "python.exe"
        wheel = wheel_for(run, interpreter, "win_amd64")

        run_logged(["py", f"-{interpreter}", "-m", "venv", str(venv)], log)
        run_logged([str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", str(wheel)], log)

        # From the run directory, so `import mila` cannot find a source tree.
        run_logged(
            [str(python), str(run.verifier), "--provenance", "--expect-version", run.pep440],
            log,
            cwd=run.directory,
            environment=environment,
        )


def stage_test_linux_wheels(run: Run, log: Path) -> None:
    # python:*-slim carries no CUDA at all, so this is the full proof: the wheel stands on its
    # own dependencies. The exact file is installed, never a name resolved from an index.
    for interpreter in INTERPRETERS:
        wheel = wheel_for(run, interpreter, "manylinux").name
        wsl(
            f"docker run --rm "
            f"-v {run.wsl_source}/out/wheel:/wheels:ro "
            f"-v {run.wsl_checks}/verify_wheel_cleanroom.py:/verify.py:ro "
            f"-w /tmp python:{interpreter}-slim "
            f"sh -c 'pip install --quiet --disable-pip-version-check --root-user-action=ignore /wheels/{wheel} "
            f"&& python /verify.py --expect-version {run.pep440}'\n",
            log,
        )


def stage_images(run: Run, log: Path) -> None:
    # One run's images at a time: the -devel image alone is over 20 GB.
    wsl(
        f"{{ docker image ls '{LOCAL_IMAGE}' --format '{{{{.Repository}}}}:{{{{.Tag}}}}' "
        f"| grep -v ':{run.short}-' || true; }} | xargs -r docker image rm\n",
        log,
    )

    # Both targets share the builder stage, so the second build reuses the first's compile.
    # MILA_CLEAN_BUILD is forced: BuildKit cache mounts survive --no-cache, and an image built
    # from another tree's objects is a failure that builds successfully.
    for target in ("runtime", "devel"):
        wsl(
            f"cd {run.wsl_source}\n"
            f"MILA_RUNTIME_IMAGE_TAG={run.image(target)} MILA_IMAGE_TARGET={target} MILA_CLEAN_BUILD=1 "
            f"bash scripts/dockerhub/build-runtime-image.sh\n",
            log,
        )


def stage_test_runtime_image(run: Run, log: Path) -> None:
    wsl(f"MILA_IMAGE={run.image('runtime')} bash {run.wsl_checks}/verify-image.sh\n", log)


def stage_test_devel_image(run: Run, log: Path) -> None:
    wsl(f"MILA_IMAGE={run.image('devel')} bash {run.wsl_checks}/verify-devel-image.sh\n", log)


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    action: Callable[[Run, Path], None]


STAGES = (
    Stage("export", "Copy the committed files out of the repository", stage_export),
    Stage("windows-wheels", "Build the Windows wheels", stage_windows_wheels),
    Stage("linux-wheels", "Build the Linux wheels", stage_linux_wheels),
    Stage("collect-wheels", "Collect the wheels: exactly four, one version", stage_collect_wheels),
    Stage("test-windows-wheels", "Install test: Windows wheels load their own CUDA", stage_test_windows_wheels),
    Stage("test-linux-wheels", "Install test: Linux wheels on a machine with no CUDA", stage_test_linux_wheels),
    Stage("images", "Build the -runtime and -devel images", stage_images),
    Stage("test-runtime-image", "Run the website's quick start against -runtime", stage_test_runtime_image),
    Stage("test-devel-image", "Walk the website's Docker tab against -devel", stage_test_devel_image),
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    return {}


def build(arguments: argparse.Namespace) -> int:
    commit = git("rev-parse", "--verify", f"{arguments.commit}^{{commit}}")
    short = commit[:12]
    version = git("show", f"{commit}:Version.txt").strip()
    workspace = Path(arguments.workspace).resolve()

    run = Run(
        commit=commit,
        short=short,
        version=version,
        pep440=pep440_version(version),
        directory=workspace / short,
        wsl_directory=f"{WSL_WORKSPACE}/{short}",
    )

    selected = [stage.name for stage in STAGES]

    if arguments.only:
        selected = [name.strip() for name in arguments.only.split(",")]
        unknown = [name for name in selected if name not in {stage.name for stage in STAGES}]

        if unknown:
            print(f"Unknown stage(s): {', '.join(unknown)}. Stages: {', '.join(stage.name for stage in STAGES)}")
            return 2

    state_path = run.directory / "state.json"

    if not arguments.resume and not arguments.only:
        remove_tree(run.directory)

    run.directory.mkdir(parents=True, exist_ok=True)
    run.logs.mkdir(exist_ok=True)
    state = load_state(state_path)
    state.update({"commit": commit, "version": version, "pep440": run.pep440})
    stages_state = state.setdefault("stages", {})

    subject = git("log", "-1", "--format=%s", commit)
    print(f"Release build  {version}  (wheels {run.pep440})")
    print(f"Commit         {short}  {subject}")
    print(f"Run directory  {run.directory}")
    print(f"Linux copy     {WSL_DISTRIBUTION}:{run.wsl_directory}")
    print()

    started = time.monotonic()
    failed_stage = None
    copy_checks(run, run.logs / "checks.log")

    for stage in STAGES:
        if stage.name not in selected:
            continue

        if arguments.resume and stages_state.get(stage.name, {}).get("result") == "pass":
            print(f"  skip  {stage.description} (passed earlier)")
            continue

        log = run.logs / f"{stage.name}.log"
        print(f"  ....  {stage.description}   [log: {log}]", flush=True)
        stage_started = time.monotonic()

        try:
            stage.action(run, log)
            result = "pass"
        except (StageFailed, subprocess.CalledProcessError, OSError) as error:
            result = "FAIL"

            with open(log, "a", encoding="utf-8") as handle:
                handle.write(f"\nSTAGE FAILED: {error}\n")

            print(f"          {error}")

        seconds = time.monotonic() - stage_started
        stages_state[stage.name] = {"result": result, "seconds": round(seconds)}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"  {result.lower() if result == 'pass' else result}  {stage.description} ({seconds / 60:.1f} min)", flush=True)

        if result != "pass":
            failed_stage = stage.name
            break

    total = time.monotonic() - started
    outcome = "pass" if failed_stage is None else f"FAIL at {failed_stage}"

    with open(workspace / "history.log", "a", encoding="utf-8") as handle:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        handle.write(f"{stamp}  {short}  {version:<16}  {outcome:<32}  {total / 60:5.1f} min  build\n")

    print()

    if failed_stage is None and arguments.only:
        print(f"The selected stages passed ({total / 60:.1f} min).")
        return 0

    if failed_stage is None:
        print(f"Everything passed ({total / 60:.1f} min). Wheels: {run.wheels}")
        print(f"Images: {run.image('runtime')}, {run.image('devel')}")
        return 0

    print(f"Stopped at '{failed_stage}' ({total / 60:.1f} min). Log: {run.logs / (failed_stage + '.log')}")
    print("Fix it, commit, and run again -- or add --resume to rerun from the failed stage on the same commit.")

    return 1


def build_passed(workspace: Path, commit: str) -> bool:
    """Whether every build stage has passed for exactly this commit."""
    state = load_state(workspace / commit[:12] / "state.json")
    stages = state.get("stages", {})

    return state.get("commit") == commit and all(
        stages.get(stage.name, {}).get("result") == "pass" for stage in STAGES)


# ---------------------------------------------------------------------------
# Releasing: prepare and finish
# ---------------------------------------------------------------------------

VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-dev\+(\d+))?")


def parse_version(text: str) -> tuple[int, int, int, int | None]:
    match = VERSION_PATTERN.fullmatch(text.strip())

    if not match:
        raise StageFailed(f"Version.txt '{text.strip()}' is neither X.Y.Z nor X.Y.Z-dev+N")

    major, minor, patch, counter = match.groups()

    return int(major), int(minor), int(patch), None if counter is None else int(counter)


def release_version(working: str, kind: str) -> str:
    """
    The version a release of this kind cuts from a working version.

    dev carries the next PATCH by default, so a patch releases what Version.txt already
    names, and a minor renames it: 0.21.1-dev+7 releases as 0.22.0. A working version that is
    already X.Y.0 names a minor, which only happens where a numbering era opened straight onto
    one -- 0.21.0-dev+N is that case.
    """
    major, minor, patch, counter = parse_version(working)

    if counter is None:
        raise StageFailed(f"Version.txt is already a release version ({working}); run `finish`.")

    if kind == "patch":
        if patch == 0:
            raise StageFailed(
                f"{working} names a minor, not a patch. `prepare minor` releases it; a patch of "
                f"{major}.{minor - 1 if minor else 0} would need a version dev does not carry.")

        return f"{major}.{minor}.{patch}"

    return f"{major}.{minor}.0" if patch == 0 else f"{major}.{minor + 1}.0"


def version_file() -> Path:
    return repository / "Version.txt"


def version_sites(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repository / "scripts" / "release" / "version_sites.py"), *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def require_on_dev() -> None:
    branch = git("symbolic-ref", "--short", "HEAD", check=False)

    if branch != "dev":
        raise StageFailed(f"releases are cut from dev; the working copy is on '{branch or 'a detached HEAD'}'.")


def tracked_changes() -> list[str]:
    status = git("status", "--porcelain", "--untracked-files=no")

    return status.splitlines() if status else []


def prepare(arguments: argparse.Namespace) -> int:
    require_on_dev()

    if tracked_changes():
        raise StageFailed("dev has uncommitted changes. Commit or set them aside before preparing a release.")

    working = version_file().read_text(encoding="utf-8").strip()
    version = release_version(working, arguments.kind)

    if git_succeeds("rev-parse", "--verify", f"refs/tags/v{version}"):
        raise StageFailed(f"v{version} already exists.")

    version_file().write_text(version + "\n", encoding="utf-8", newline="\n")
    print(f"Version.txt    {working} -> {version}")

    if arguments.kind == "patch":
        print()
        print("A patch is a tag only: no wheels, no images, no site, and no version sites move.")
        print("Version.txt is the only change. Review it, then run:")
        print("  python scripts/release/release.py finish")
        return 0

    # The sixteen hand-written sites a reader copies. --expect proves they now name this release.
    for step in (("--set", version), ("--check", "--expect")):
        result = version_sites(*step)

        if result.returncode != 0:
            print(result.stdout + result.stderr)
            raise StageFailed(f"version_sites.py {' '.join(step)} failed")

    print(f"Version sites  all sixteen set to {version}, and checked")

    audit = version_sites("--audit-prose")
    print()
    print("Prose this release may have made false -- rewrite, never bump:")
    print("\n".join(f"  {line}" for line in audit.stdout.strip().splitlines()))
    print()
    print("Also: reconcile BACKLOG.md and ROADMAP.md, and clear any 'not published yet' copy.")
    print()
    print("Nothing is committed. Review the diff in Visual Studio, make the prose edits, then run:")
    print("  python scripts/release/release.py finish")

    return 0


def confirm(version: str, actions: list[str]) -> bool:
    print()
    print("About to, in the local repository only:")

    for action in actions:
        print(f"  - {action}")

    answer = input(f"Type the version ({version}) to go ahead: ").strip()

    return answer == version


def finish(arguments: argparse.Namespace) -> int:
    require_on_dev()
    version = version_file().read_text(encoding="utf-8").strip()
    major, minor, patch, counter = parse_version(version)

    if counter is not None:
        raise StageFailed(f"Version.txt is {version}, a working version. Run `prepare minor` or `prepare patch` first.")

    kind = "minor" if patch == 0 else "patch"
    tag = f"v{version}"
    next_version = f"{major}.{minor}.{patch + 1}-dev+1"

    if git_succeeds("rev-parse", "--verify", f"refs/tags/{tag}"):
        raise StageFailed(f"{tag} already exists.")

    if kind == "minor":
        result = version_sites("--check", "--expect")

        if result.returncode != 0:
            print(result.stdout + result.stderr)
            raise StageFailed("the version sites do not all name this release")

    # Before anything is committed, so a refusal leaves dev as it was. Nothing but a release
    # merge ever moves master, so the two copies differ only by releases one side has not seen
    # yet: the local one lags when a release was merged elsewhere, and leads while releases are
    # cut locally and not pushed. Either way the newer one is the base; only a genuine
    # divergence stops the release.
    git("fetch", "--quiet", "origin", "master")
    remote_master = git("rev-parse", "refs/remotes/origin/master")
    local_master = git("rev-parse", "refs/heads/master")

    if git_succeeds("merge-base", "--is-ancestor", local_master, remote_master):
        base = remote_master
    elif git_succeeds("merge-base", "--is-ancestor", remote_master, local_master):
        base = local_master
    else:
        raise StageFailed("local master and origin/master have diverged. Resolve that by hand first.")

    # 1. The release-prep commit. Taken as-is if it was already committed by hand.
    if tracked_changes():
        print(f"Committing the release-prep changes on dev ({len(tracked_changes())} file(s)).")
        git("commit", "--all", "--quiet", "-m", f"Version: {version}\nRelease {version}")

    prep_commit = git("rev-parse", "HEAD")

    if git("show", f"{prep_commit}:Version.txt").strip() != version:
        raise StageFailed("HEAD does not carry the release version in Version.txt.")

    print(f"Release commit {prep_commit[:12]}  {git('log', '-1', '--format=%s', prep_commit)}")

    # 2. Everything the release publishes, built and tested from that commit. A patch
    # publishes nothing, so there is nothing of this kind to build.
    workspace = Path(arguments.workspace).resolve()

    if kind == "minor":
        if build_passed(workspace, prep_commit):
            print("Release build  already passed for this commit")
        else:
            print()
            status = build(argparse.Namespace(commit=prep_commit, workspace=str(workspace), resume=True, only=None))

            if status != 0:
                print()
                print("The release build failed, so nothing was merged or tagged. The prep commit")
                print("stays on dev; fix, commit, and run `finish` again.")
                return 1
    else:
        print("Release build  none -- a patch publishes no files")

    # 3. The merge onto master, written without a checkout.
    merged_tree = git("merge-tree", "--write-tree", base, prep_commit, check=False).splitlines()[0]

    # dev contains every master commit's content, so the merge is exactly dev's tree. Anything
    # else means master holds a change dev does not, and releasing would ship it untested.
    if merged_tree != git("rev-parse", f"{prep_commit}^{{tree}}"):
        raise StageFailed("merging dev into master does not give dev's exact files; master holds a change dev lacks.")

    actions = [
        f"merge dev ({prep_commit[:12]}) into master",
        f"tag the merge {tag}",
        f"set Version.txt on dev to {next_version} and commit it",
    ]

    if not confirm(version, actions):
        print("Not confirmed. The prep commit is on dev; nothing else changed.")
        return 1

    merge = git("commit-tree", merged_tree, "-p", base, "-p", prep_commit, "-m", f"Merge dev for release {version}")
    git("update-ref", "-m", f"release {version}", "refs/heads/master", merge, local_master)
    git("tag", tag, merge)
    print(f"master         {local_master[:12]} -> {merge[:12]}")
    print(f"tag            {tag} -> {merge[:12]}")

    # 4. Open the next version on dev. The next patch by default: a tag needs no decision.
    version_file().write_text(next_version + "\n", encoding="utf-8", newline="\n")
    git("commit", "--quiet", "-m", f"Version: {next_version}\nOpen {next_version}", "--", "Version.txt")
    print(f"dev            opened {next_version} at {git('rev-parse', '--short=12', 'HEAD')}")

    with open(workspace / "history.log", "a", encoding="utf-8") as handle:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        handle.write(f"{stamp}  {merge[:12]}  {version:<16}  {'tagged ' + tag + ' locally':<32}  {'':>9}  finish\n")

    print()
    print("Done, locally. NOTHING WAS PUSHED: origin still has the old master and dev, and no")
    print(f"{tag}. Publishing is not part of this version of the script.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build", help="build and verify the release files for one commit")
    build_parser.add_argument("--commit", default="HEAD", help="the commit to build (default HEAD)")
    build_parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help=f"default {DEFAULT_WORKSPACE}")
    build_parser.add_argument("--resume", action="store_true", help="skip stages that already passed for this commit")
    build_parser.add_argument("--only", help="run only these stages, comma-separated")

    prepare_parser = commands.add_parser("prepare", help="set the release version on dev, uncommitted")
    prepare_parser.add_argument("kind", choices=("minor", "patch"))

    finish_parser = commands.add_parser("finish", help="commit, build, confirm, merge, tag, reopen dev")
    finish_parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help=f"default {DEFAULT_WORKSPACE}")

    for command_parser in (build_parser, prepare_parser, finish_parser):
        command_parser.add_argument("--repository", help=argparse.SUPPRESS)

    arguments = parser.parse_args()

    if arguments.repository:
        global repository
        repository = Path(arguments.repository).resolve()

    try:
        if arguments.command == "build":
            return build(arguments)

        if arguments.command == "prepare":
            return prepare(arguments)

        return finish(arguments)
    except StageFailed as error:
        print(f"\nStopped: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
