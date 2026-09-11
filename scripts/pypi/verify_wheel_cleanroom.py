# Verify an INSTALLED mila-llm wheel on a machine with no CUDA Toolkit.
#
# The question this answers is not "does the wheel import" -- that passes trivially on
# any developer machine, because a machine with a Toolkit resolves the CUDA libraries
# whether or not the wheel's own dependencies would have. Three environments passed the
# Linux wheel before a CUDA-free image found it was missing nvidia-cuda-runtime
# entirely. So this script ASSERTS THE ABSENCE first and refuses to report success from
# a contaminated environment: a green run here means the wheel stood on its own.
#
# It deliberately does not need a GPU. The failure mode being gated is library
# resolution, which happens when the extension is loaded, long before a device is
# touched -- CudaDeviceRegistrar treats a device count of zero as a warning, so
# initialize() succeeds on a GPU-less runner and is a legitimate assertion here.
#
# Run it from a directory that is not the repository, so `import mila` cannot find the
# source tree instead of the installed distribution. The site-packages assertion below
# catches that anyway rather than trusting the caller to get it right.

import argparse
import os
import sys
import sysconfig
from pathlib import Path

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> bool:
    """Record one assertion. Returns the condition so a caller can branch on it."""
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")

        if detail:
            print(f"        {detail}")

        failures.append(description)

    return condition


def site_package_roots() -> list[Path]:
    """Directories an installed distribution may legitimately live in."""
    roots = []

    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)

        if path:
            roots.append(Path(path).resolve())

    return roots


def is_inside(candidate: Path, roots: list[Path]) -> bool:
    resolved = candidate.resolve()

    return any(resolved == root or root in resolved.parents for root in roots)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--expect-version",
    help="Exact version the installed distribution must report. TestPyPI and PyPI both "
         "carry mila-llm, and an older PyPI release outranks a newer TestPyPI dev build "
         "in some resolutions -- asserting the version is what proves which one arrived.")
args = parser.parse_args()

print(f"Python      {sys.version.split()[0]} ({sys.executable})")
print(f"Platform    {sys.platform}")
print()

# ---------------------------------------------------------------------------
# 1. The clean room itself. Checked BEFORE the import, because every assertion
#    after this one is meaningless if a Toolkit is present.
# ---------------------------------------------------------------------------

print("Clean room")

if os.name == "nt":
    cuda_path = os.environ.get("CUDA_PATH")
    default_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")

    check(not cuda_path,
          "CUDA_PATH is unset",
          f"CUDA_PATH={cuda_path}")
    check(not default_root.is_dir(),
          "no CUDA Toolkit at the default install root",
          f"{default_root} exists")
else:
    # The Linux extension resolves libcudart.so.13 and friends through the normal
    # loader path, so a system CUDA registered with ldconfig would mask a missing
    # dependency exactly the way an installed Toolkit does on Windows.
    system_roots = [Path("/usr/local/cuda"), Path("/usr/local/cuda-13")]
    present = [str(root) for root in system_roots if root.exists()]

    check(not present,
          "no system CUDA installation",
          f"found {present}")

contaminated = bool(failures)

if contaminated:
    print()
    print("This environment is NOT a clean room, so nothing below can establish the")
    print("property this script exists to check -- a wheel leaning on a host Toolkit")
    print("passes here exactly the way a correct one does. Continuing for diagnosis;")
    print("the result is invalid regardless of what follows.")

print()

# ---------------------------------------------------------------------------
# 2. The import. This is where the extension's library dependencies resolve, and
#    it is the whole point of the exercise.
# ---------------------------------------------------------------------------

print("Import")

try:
    import mila
except ModuleNotFoundError as error:
    # Distinguished from the case below on purpose: the package not being installed
    # is a mistake in how the script was invoked, not a finding about the wheel.
    if error.name == "mila":
        print("  FAIL  import mila")
        print(f"        {error}")
        print()
        print("mila-llm is not installed in this interpreter. Install the wheel first;")
        print("this script tests an installed distribution, it does not build one.")
        sys.exit(2)

    raise
except Exception as error:
    print("  FAIL  import mila")
    print(f"        {type(error).__name__}: {error}")
    print()
    print("The package is installed but its extension could not load. On a clean machine")
    print("this is normally a CUDA library the wheel's dependencies do not pin -- compare")
    print("the directories the error names against the nvidia-* packages pip installed.")
    sys.exit(1)

check(True, "import mila")

roots = site_package_roots()
module_path = Path(mila.__file__)

check(is_inside(module_path, roots),
      "mila was imported from site-packages, not a source tree",
      f"{module_path} is outside {[str(root) for root in roots]}")

if args.expect_version:
    check(mila.__version__ == args.expect_version,
          f"installed version is {args.expect_version}",
          f"got {mila.__version__}")
else:
    notes.append(f"version not asserted (installed: {mila.__version__})")

print()

# ---------------------------------------------------------------------------
# 3. Where the CUDA libraries came from. On a clean machine every one of them must
#    have come from the wheel's own dependencies; anything else means the wheel is
#    leaning on the host and the gate has not actually been tested.
# ---------------------------------------------------------------------------

print("CUDA library provenance")

directories = [Path(directory) for directory in mila.cuda_library_directories]

for directory in directories:
    print(f"        {directory}")

if check(bool(directories),
         "at least one CUDA library directory was located",
         "none were found -- the nvidia-* dependencies are missing"):

    outside = [str(directory) for directory in directories if not is_inside(directory, roots)]

    check(not outside,
          "every registered CUDA directory is in site-packages",
          f"host-supplied: {outside}")

# On Windows the check above has teeth, because cuda_library_directories includes any
# Toolkit directory that was found. On LINUX it does not: that list is only ever the
# wheel's own directories, so it passes by construction and proves nothing. What can
# actually go wrong on Linux is different -- the extension's DT_NEEDED entries resolving
# through ldconfig to a system library instead of the preloaded wheel copy. That is only
# visible after the fact, in what the process actually has mapped.

if sys.platform.startswith("linux"):
    print()
    print("Loaded CUDA libraries (Linux: what the loader actually resolved)")

    # The driver's libraries are deliberately absent from this list. libcuda and
    # libnvidia-* come from the host by definition -- that is what a driver is, and
    # under `docker run --gpus all` they are injected into the container. Only the
    # runtime and math libraries are the wheel's responsibility.
    vendored_prefixes = (
        "libcudart", "libcublas", "libcurand", "libnvrtc", "libnvJitLink",
    )

    mapped: set[Path] = set()

    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            fields = line.rstrip("\n").split(maxsplit=5)

            if len(fields) == 6 and fields[5].startswith("/") and ".so" in fields[5]:
                mapped.add(Path(fields[5]))

    vendored = sorted(
        path for path in mapped
        if path.name.startswith(vendored_prefixes))

    for path in vendored:
        origin = "wheel" if is_inside(path, roots) else "HOST"
        print(f"        [{origin}] {path}")

    # Assert the observation happened before asserting anything about it. A wheel that
    # loaded no CUDA runtime library at all would otherwise sail through the next check
    # on an empty set -- the same vacuous pass this whole block exists to remove.
    if check(bool(vendored),
             "the CUDA runtime libraries are mapped into the process",
             "none observed -- this check established nothing; investigate before trusting a pass"):

        host_supplied = [str(path) for path in vendored if not is_inside(path, roots)]

        check(not host_supplied,
              "every CUDA runtime library was resolved from site-packages",
              f"resolved from the host instead: {host_supplied}")

print()

# ---------------------------------------------------------------------------
# 4. Framework initialization. No GPU required: a device count of zero is a warning
#    inside CudaDeviceRegistrar, not an error, so this is a fair assertion here.
# ---------------------------------------------------------------------------

print("Initialization")

try:
    mila.initialize("warning")
    check(True, "mila.initialize() completed")
except Exception as error:
    check(False, "mila.initialize() completed", f"{type(error).__name__}: {error}")

print()

# ---------------------------------------------------------------------------

for note in notes:
    print(f"  NOTE  {note}")

if failures:
    print()
    print(f"FAILED -- {len(failures)} check(s):")

    for failure in failures:
        print(f"  - {failure}")

    if contaminated:
        print()
        print("Note the first failures are the environment, not the wheel. Re-run where")
        print("no CUDA Toolkit is installed; a developer machine cannot answer this.")

    sys.exit(1)

print("OK -- the wheel stands on its own dependencies.")
