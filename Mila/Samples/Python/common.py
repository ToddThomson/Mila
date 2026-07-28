"""
Shared plumbing for the Mila Python samples: find the built extension, find the
weights, load a model.

Nothing here is inference. It exists so chat.py and generate.py can open with the
part you came to read. Standard library only -- there is no requirements.txt for
these samples, and that is deliberate.
"""

import os
import sys
import sysconfig
from pathlib import Path

# .../Mila/Samples/Python/common.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Where the converter writes Mila binary artifacts. Both files are produced by
# Tools/Converters (see Data/Models/README.md); neither is redistributed here.
MODELS_DIR = REPO_ROOT / "Data" / "Models"

MODEL_CATALOG = {
    "gemma": {
        "weights": MODELS_DIR / "Gemma" / "gemma4_12b_it_bf16.bin",
        "tokenizer": MODELS_DIR / "Gemma" / "gemma_tokenizer.bin",
        "context_length": 4096,
    },
    "llama": {
        "weights": MODELS_DIR / "LLaMa" / "llama32_3b_instruct_bf16.bin",
        "tokenizer": MODELS_DIR / "LLaMa" / "llama32_tokenizer.bin",
        "context_length": 4096,
    },
}


def configure_console():
    """
    Make stdout able to carry model output.

    A legacy Windows console is cp1252, and a tokenizer piece or an em dash from
    the model raises UnicodeEncodeError mid-response. UTF-8 with replacement is the
    difference between a wrong glyph and a crashed sample.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Locating the extension module
# ---------------------------------------------------------------------------

def _extension_name():
    """
    The extension filename this interpreter can load.

    EXT_SUFFIX carries the ABI tag (e.g. .cp313-win_amd64.pyd). A build made for a
    different Python cannot be imported, so the tag is the whole point of matching
    on it rather than on '*.pyd'.
    """
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if os.name == "nt" else ".so")

    return "_mila" + suffix


def _search_dirs():
    """Directories that may hold the mila package, most specific first."""
    override = os.environ.get("MILA_PYD_DIR")

    if override:
        yield Path(override)

    # The neutral build output: <build dir>/python, published by
    # Mila/Bindings/CMakeLists.txt. Newest build wins when several presets exist.
    build_dirs = sorted(
        REPO_ROOT.glob("out/build/*/python"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    yield from build_dirs

    # The Inference Server's convenience copy.
    yield REPO_ROOT / "Mila" / "Adaptors" / "Inference" / "Server"


def _find_extension_dir():
    """
    Find a directory to put on sys.path such that `import mila` resolves.

    That is the directory CONTAINING the package, not the package itself -- the
    extension is mila/_mila<abi>.pyd and the __init__ beside it is what registers
    the CUDA DLL directories before the extension loads.
    """
    name = _extension_name()
    mismatched = []

    for directory in _search_dirs():
        package = directory / "mila"

        if not package.is_dir():
            continue

        if (package / name).is_file() and (package / "__init__.py").is_file():
            return directory

        mismatched.extend(sorted(package.glob("_mila*.pyd")) + sorted(package.glob("_mila*.so")))

    if mismatched:
        found = "\n  ".join(str(path.name) for path in mismatched)
        raise ImportError(
            f"Found a mila extension, but not one this interpreter can load.\n"
            f"  This interpreter: Python {sys.version.split()[0]}, expects '{name}'\n"
            f"  Found:\n  {found}\n"
            f"Rebuild MilaPy against this Python, or run the samples with the Python "
            f"the extension was built for."
        )

    raise ImportError(
        "Could not find the mila package.\n"
        "Either `pip install mila-llm`, or build the MilaPy target (which publishes "
        "the package to <build dir>/python), or set MILA_PYD_DIR to the directory "
        "containing it."
    )


def import_mila(log_level="warning"):
    """
    Import and initialize mila.

    An installed wheel wins; failing that the package is located in the build tree.
    Either way the CUDA DLL directories are registered by mila/__init__.py before
    the extension loads -- that is the package's job, not the sample's.

    Returns the module. log_level is one of trace | info | warning | error and is
    what mila.initialize() takes; 'info' prints load progress and timings.
    """
    try:
        import mila  # installed wheel, or already on PYTHONPATH
    except ImportError:
        sys.path.insert(0, str(_find_extension_dir()))
        import mila

    mila.initialize(log_level)

    return mila


# ---------------------------------------------------------------------------
# Locating the weights
# ---------------------------------------------------------------------------

def resolve_paths(family, weights=None, tokenizer=None):
    """
    Resolve (weights, tokenizer) for a model family, with explicit paths and then
    the MILA_MODEL_PATH / MILA_TOKENIZER_PATH environment variables taking
    precedence over the catalog defaults.
    """
    entry = MODEL_CATALOG[family]

    weights_path = Path(weights or os.environ.get("MILA_MODEL_PATH") or entry["weights"])
    tokenizer_path = Path(tokenizer or os.environ.get("MILA_TOKENIZER_PATH") or entry["tokenizer"])

    for label, path in (("weights", weights_path), ("tokenizer", tokenizer_path)):
        if not path.is_file():
            raise FileNotFoundError(
                f"Mila {family} {label} not found at {path}.\n"
                "Mila does not ship model weights. Convert a checkpoint with "
                "Tools/Converters (see Data/Models/README.md), then pass --weights / "
                "--tokenizer or set MILA_MODEL_PATH / MILA_TOKENIZER_PATH."
            )

    return weights_path, tokenizer_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(mila, family, weights, tokenizer, context_length, device_index=0, quantize_fp8=False):
    """
    Load a tokenizer and model pair for the given family.

    Gemma is loaded with FP4 weights (the binding hardcodes that -- 12B only fits a
    12 GB card quantized); Llama is BF16 unless quantize_fp8 is set.
    """
    if family == "gemma":
        return (
            mila.BpeTokenizer.load_gemma(str(tokenizer)),
            mila.GemmaModel.from_pretrained(str(weights), context_length, device_index),
        )

    return (
        mila.BpeTokenizer.load_llama32(str(tokenizer)),
        mila.LlamaModel.from_pretrained(str(weights), context_length, device_index, quantize_fp8),
    )
