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
    "cuda_dll_directories",
]


def _register_cuda_dll_directories() -> list[str]:
    """
    Make the CUDA runtime DLLs findable, before the extension is loaded.

    Since Python 3.8, Windows does NOT search PATH when resolving an extension
    module's DLL dependencies -- only system directories, the extension's own
    directory, and directories passed to os.add_dll_directory. The extension links
    cuBLASLt and cuRAND, so without this a machine with a perfectly good CUDA
    install on PATH still fails with a bare "DLL load failed", a message that names
    neither the missing library nor the reason.

    Two sources. NVIDIA's own PyPI wheels (nvidia-cublas, nvidia-curand) are
    registered first, since they are what this package's dependencies pinned. An
    installed CUDA Toolkit is registered after them as a backstop -- deliberately as
    well as, not instead of: a partial wheel install (one library present, another
    missing) would otherwise fail to load while a perfectly good toolkit sat unused.
    With both dependencies installed the toolkit is never reached.

    Returns the directories registered. Empty off Windows, where the loader's normal
    search path resolves the .so.
    """
    if _os.name != "nt":
        return []

    registered: list[str] = []

    for directory in _nvidia_wheel_directories() + _toolkit_directories():
        _os.add_dll_directory(str(directory))
        registered.append(str(directory))

    return registered


def _nvidia_wheel_directories() -> list[_Path]:
    """
    CUDA DLL directories provided by NVIDIA's PyPI wheels, if installed.

    Located by finding the DLLs themselves rather than by assuming a directory
    convention, because NVIDIA has changed it: the CUDA 12 wheels used
    nvidia/<library>/bin, CUDA 13 uses nvidia/cu13/bin/x86_64. Measured, not
    guessed -- the first version of this globbed nvidia/*/bin, which on a real
    install matched a directory holding only the x86_64 subdirectory and no DLLs at
    all. Searching for *.dll survives the next reorganisation and the aarch64 split.
    """
    nvidia_root = _Path(__file__).resolve().parent.parent / "nvidia"

    if not nvidia_root.is_dir():
        return []

    return sorted({dll.parent for dll in nvidia_root.rglob("*.dll")})


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


#: CUDA directories registered at import time. Print this when a load fails --
#: an empty list on Windows means no CUDA was found at all.
cuda_dll_directories = _register_cuda_dll_directories()

try:
    from ._mila import (
        BpeTokenizer,
        GemmaModel,
        LlamaModel,
        StopController,
        initialize,
    )
except ImportError as error:
    raise ImportError(
        f"{error}\n\n"
        f"The Mila extension failed to load. CUDA directories registered: "
        f"{cuda_dll_directories or 'none'}.\n"
        "An empty list on Windows means neither NVIDIA's CUDA wheels nor a CUDA "
        "Toolkit was found; install the toolkit, or set CUDA_PATH to it."
    ) from error

def _resolve_version() -> str:
    """The installed distribution's version, or a marker when run from a build tree."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mila-llm")
    except PackageNotFoundError:
        return "0.0.0+local"


__version__ = _resolve_version()
