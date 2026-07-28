"""
Makes the CUDA runtime DLLs findable before `import mila`.

Importing this module registers the CUDA Toolkit's binary directories; import it
ahead of `mila` in any module that imports the binding.

Why it is needed at all: since Python 3.8, Windows does NOT search PATH when
resolving an extension module's DLL dependencies. The search is limited to system
directories, the extension's own directory, and directories passed to
os.add_dll_directory. The mila binding links cublasLt and curand dynamically and
those live in the toolkit, so a machine with a perfectly good CUDA install on PATH
still fails with a bare "DLL load failed while importing mila" -- a message that
names neither the missing DLL nor the reason.

Layout differs by toolkit generation: CUDA 13 puts the redistributable DLLs in
bin\\x64, earlier versions keep them directly in bin. Both are registered when
present, so this does not have to be revisited on a toolkit upgrade, and a
directory that does not exist is skipped rather than raising (os.add_dll_directory
raises FileNotFoundError on a missing path).

No-op off Windows: Linux resolves the .so's dependencies through the normal
loader search path.
"""
import os
from pathlib import Path

# The default Windows install root, used when CUDA_PATH is unset or points at a
# toolkit that no longer exists. Newest version first.
_DEFAULT_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")

    if cuda_path:
        roots.append(Path(cuda_path))

    if _DEFAULT_ROOT.is_dir():
        roots.extend(sorted(_DEFAULT_ROOT.iterdir(), reverse=True))

    return roots


def register_cuda_dll_directories() -> list[str]:
    """
    Register the binary directories of ONE CUDA toolkit: the first candidate root
    that has any -- CUDA_PATH when it is set and real, otherwise the newest
    installed. Registering every installed toolkit would leave which copy of a DLL
    wins up to the loader's ordering across the directory list.

    Returns the directories registered, which is empty off Windows and is the thing
    to print when diagnosing a load failure.
    """
    if os.name != "nt":
        return []

    for root in _candidate_roots():
        directories = [
            str(directory)
            for directory in (root / "bin" / "x64", root / "bin")
            if directory.is_dir()
        ]

        if directories:
            for directory in directories:
                os.add_dll_directory(directory)

            return directories

    return []


CUDA_DLL_DIRECTORIES = register_cuda_dll_directories()
