"""
Mila -- a C++23/CUDA runtime for open large language models.

Read the loop, don't just call it: every forward pass is explicit, device and
precision are compile-time decisions, and there is no hidden execution engine.
This package is the Python projection of that runtime.

    import mila

    mila.initialize("warning")
    tokenizer = mila.BpeTokenizer.load_gemma("gemma_tokenizer.bin")
    model = mila.GemmaModel.from_pretrained("gemma4_12b_it_bf16.bin", 4096)

    model.generate_streaming(tokenizer.encode(prompt), print)

The GIL is released around generation, so a streaming callback runs on a live
interpreter and StopController cancels a decode loop already in flight.

Source and documentation: https://github.com/toddthomson/Mila
"""

# Imported under private names so they do not show up in dir(mila) or tab
# completion: this module IS the public API, and everything visible in it reads as
# part of that API whether __all__ lists it or not.
import os as _os
from pathlib import Path as _Path

__all__ = [
    "initialize",
    "BpeTokenizer",
    "GemmaModel",
    "LlamaModel",
    "StopController",
    "cuda_library_directories",
    # Distribution: the local store, shared with Chat and the Inference Server.
    "ModelStore",
    "StoredModel",
    "StoreUsage",
    "RemovalReport",
    "HubModel",
    "HttpResponse",
    "default_hub_owner",
]


def _register_cuda_libraries() -> list[str]:
    """
    Make the CUDA runtime libraries loadable, before the extension is imported.

    Both platforms need help, for different reasons, and the same answer works for
    both: LOAD the libraries this package's dependencies pinned, rather than merely
    making them findable. A dependency is resolved against what is already in the
    process -- by base name on Windows, by SONAME on Linux -- so a library loaded
    here is the one the extension gets.

    On WINDOWS, PATH is not searched at all when resolving an extension module's DLL
    dependencies (Python 3.8 narrowed it to system directories, the extension's own
    directory, and os.add_dll_directory). Directory registration alone is also not
    enough to make the wheel's copies win: added directories are searched in an
    UNSPECIFIED order, and measurement showed a machine's CUDA v13.3 shadowing the
    site-packages copy -- a wheel silently binding to whatever toolkit is installed,
    which is what the dependency pins exist to prevent.

    On LINUX the extension's DT_NEEDED entries are resolved through the normal loader
    search path, which does NOT include site-packages. Without a system CUDA
    registered with ldconfig -- exactly the machine a wheel exists to serve -- the
    import fails on a missing libcublasLt.so. Preloading with RTLD_GLOBAL satisfies
    those entries by SONAME.

    An installed CUDA Toolkit remains a backstop on Windows, so that a partial wheel
    install degrades instead of failing while a perfectly good toolkit sits unused.

    Returns the directories the libraries came from -- print it when a load fails.
    """
    wheel_directories = _nvidia_wheel_directories()

    if _os.name == "nt":
        registered = wheel_directories + _toolkit_directories()

        # Register first: a preloaded library may itself pull in siblings.
        for directory in registered:
            _os.add_dll_directory(str(directory))
    else:
        registered = wheel_directories

    _preload(wheel_directories)

    return [str(directory) for directory in registered]


def _preload(directories: "list[_Path]") -> None:
    """
    Load every CUDA library in the given directories, making those copies
    authoritative for anything loaded afterwards.

    Best effort by design: a library that fails to load is skipped rather than raised
    on, because it may be one the extension never needs, and the real error -- with
    real context -- belongs to the extension import below.

    Two passes, because these libraries depend on each other and the filesystem order
    is not a topological one: a library that failed on the first pass because a
    sibling was not yet loaded succeeds on the second. Loading is idempotent, so a
    repeat costs a refcount rather than a second mapping.
    """
    import ctypes

    for _ in range(2):
        for directory in directories:
            for library in sorted(directory.glob("*.dll" if _os.name == "nt" else "*.so*")):
                try:
                    if _os.name == "nt":
                        ctypes.WinDLL(str(library))
                    else:
                        # RTLD_GLOBAL: the point is to satisfy the extension's
                        # DT_NEEDED entries, which a local-scope load would not.
                        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue


def _nvidia_wheel_directories() -> list[_Path]:
    """
    CUDA library directories provided by NVIDIA's PyPI wheels, if installed.

    Located by finding the libraries themselves rather than by assuming a directory
    convention, because NVIDIA has changed it: the CUDA 12 wheels used
    nvidia/<library>/{bin,lib}, CUDA 13 uses nvidia/cu13/bin/x86_64. Measured, not
    guessed -- the first version of this globbed nvidia/*/bin, which on a real
    install matched a directory holding only the x86_64 subdirectory and no libraries
    at all. Searching for the files survives the next reorganisation, the lib/bin
    split across platforms, and the aarch64 layout.
    """
    nvidia_root = _Path(__file__).resolve().parent.parent / "nvidia"

    if not nvidia_root.is_dir():
        return []

    pattern = "*.dll" if _os.name == "nt" else "*.so*"

    return sorted({library.parent for library in nvidia_root.rglob(pattern)})


def _toolkit_directories() -> list[_Path]:
    """
    Binary directories of ONE installed CUDA Toolkit: CUDA_PATH when it is set and
    real, otherwise the newest under the default install root. One rather than all,
    so it is deterministic which copy of a DLL the loader can find.
    """
    roots: list[_Path] = []
    cuda_path = _os.environ.get("CUDA_PATH")

    if cuda_path:
        roots.append(_Path(cuda_path))

    default_root = _Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")

    if default_root.is_dir():
        roots.extend(sorted(default_root.iterdir(), reverse=True))

    for root in roots:
        # CUDA 13 puts the redistributables in bin/x64; earlier layouts use bin.
        directories = [
            directory for directory in (root / "bin" / "x64", root / "bin")
            if directory.is_dir()
        ]

        if directories:
            return directories

    return []


#: CUDA library directories used at import time. Print this when a load fails --
#: an empty list means no CUDA libraries were found anywhere.
cuda_library_directories = _register_cuda_libraries()

try:
    from ._mila import (
        BpeTokenizer,
        GemmaModel,
        HttpResponse,
        HubModel,
        LlamaModel,
        ModelStore,
        RemovalReport,
        StopController,
        StoredModel,
        StoreUsage,
        default_hub_owner,
        initialize,
    )
except ImportError as error:
    raise ImportError(
        f"{error}\n\n"
        f"The Mila extension failed to load. CUDA libraries were taken from: "
        f"{cuda_library_directories or 'nowhere -- none were found'}.\n"
        "An empty list means neither NVIDIA's CUDA wheels (nvidia-cublas, "
        "nvidia-curand) nor a local CUDA Toolkit was located."
    ) from error

def _resolve_version() -> str:
    """The installed distribution's version, or a marker when run from a build tree."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mila-llm")
    except PackageNotFoundError:
        return "0.0.0+local"


__version__ = _resolve_version()
