"""
Terminal and ops: a build script that fails three ways at once -- an unset variable under `set -u`, a helper without
its execute bit, and the helper's Windows line endings. The check changes the version first, so a build that
hard-codes the output does not pass.
"""

import os
import shutil
import stat
from pathlib import Path

from hermes_tasks.common import Task, not_blocked, run

PROMPT = (
    "`./build.sh` is supposed to build dist/app.txt from src/app.txt.in, but it fails. Make it work, so that "
    "running ./build.sh from this directory in a fresh shell, with nothing set beforehand, builds dist/app.txt."
)

BUILD = """#!/usr/bin/env bash
# Builds $OUTPUT_DIR/app.txt from the template, stamped with the version.
set -euo pipefail

mkdir -p "$OUTPUT_DIR"
version=$(./tools/gen_version.sh)
sed "s/@VERSION@/$version/" src/app.txt.in > "$OUTPUT_DIR/app.txt"
echo "built $OUTPUT_DIR/app.txt ($version)"
"""

GEN_VERSION = "#!/usr/bin/env bash\n# Prints the release version.\nset -euo pipefail\n\ntr -d '\\n' < VERSION\n"

TEMPLATE = "Courier service, build @VERSION@\n"

README = """# Courier service

Release builds go to `dist/`. CI sets `OUTPUT_DIR=dist` before calling `./build.sh`.
"""


def build(workdir: Path) -> None:
    (workdir / "tools").mkdir()
    (workdir / "src").mkdir()

    script = workdir / "build.sh"
    script.write_bytes(BUILD.encode())
    script.chmod(0o755)

    helper = workdir / "tools" / "gen_version.sh"
    helper.write_bytes(GEN_VERSION.replace("\n", "\r\n").encode())
    helper.chmod(0o644)

    (workdir / "VERSION").write_bytes(b"2.7.1\n")
    (workdir / "src" / "app.txt.in").write_bytes(TEMPLATE.encode())
    (workdir / "README.md").write_bytes(README.encode())


def solve(workdir: Path) -> str:
    script = workdir / "build.sh"
    script.write_text(script.read_text().replace('set -euo pipefail\n', 'set -euo pipefail\n\nOUTPUT_DIR="${OUTPUT_DIR:-dist}"\n'))

    helper = workdir / "tools" / "gen_version.sh"
    helper.write_bytes(helper.read_bytes().replace(b"\r\n", b"\n"))
    helper.chmod(0o755)

    return "Defaulted OUTPUT_DIR to dist, made gen_version.sh executable and converted it to LF."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    shutil.rmtree(workdir / "dist", ignore_errors=True)
    (workdir / "VERSION").write_bytes(b"2.8.0\n")

    script = workdir / "build.sh"

    if not script.is_file() or not script.stat().st_mode & stat.S_IXUSR:
        return False, "build.sh is missing or not executable"

    clean = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
    result = run(["./build.sh"], workdir, env=clean)

    if result.returncode != 0:
        return False, f"exit {result.returncode}: {(result.stderr or result.stdout).strip().splitlines()[-1:]}"

    built = workdir / "dist" / "app.txt"

    if not built.is_file():
        return False, "no dist/app.txt"

    content = built.read_text()

    return content == "Courier service, build 2.8.0\n", f"dist/app.txt: {content!r}"


TASK = Task("t10_build_script", "ops", PROMPT, build, solve, check, short=True)
