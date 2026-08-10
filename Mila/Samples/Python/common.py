"""
Shared plumbing for the Mila Python samples: find the built extension, resolve a
model, load it.

Nothing here is inference. It exists so chat.py and generate.py can open with the
part you came to read. Standard library only -- there is no requirements.txt for
these samples, and that is deliberate.

There are two ways to name a model, and they are not equivalent. A store name
(load_from_store) is the normal one: the artifact is already quantized and its
record says to what. Loose .bin paths (resolve_paths + load) are the fallback for
a checkpoint converted locally from a family Mila does not publish.
"""

import os
import sys
import sysconfig
from pathlib import Path

# .../Mila/Samples/Python/common.py -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# The published model the samples open by default. Fetch it with ModelStore.pull or
# /install; nothing in these samples downloads anything.
DEFAULT_MODEL = "gemma-4-12b-it-fp4"

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


def _extension_mtime(directory):
    """
    When the extension under <directory>/mila was last built. 0 if there is none.

    Deliberately not the directory's own mtime: a rebuild rewrites the extension in
    place and leaves every enclosing directory untouched, so a preset built minutes
    ago can look older than one merely configured days ago -- and the sample then
    imports the stale one and fails on a symbol that exists in the newer build.
    """
    stamps = [
        path.stat().st_mtime
        for path in (directory / "mila").glob("_mila*")
        if path.is_file()
    ]

    return max(stamps, default=0.0)


def _search_dirs():
    """Directories that may hold the mila package, most specific first."""
    override = os.environ.get("MILA_PYD_DIR")

    if override:
        yield Path(override)

    # The neutral build output: <build dir>/python, published by
    # Mila/Bindings/CMakeLists.txt. Newest build wins when several presets exist.
    build_dirs = sorted(
        REPO_ROOT.glob("out/build/*/python"),
        key=_extension_mtime,
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
        # Full paths, not names: every candidate carries the same ABI tag, so the
        # filename alone says nothing about which build tree it came from.
        found = "\n  ".join(str(path) for path in mismatched)

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
                "Loose .bin files are the fallback path. For a published model, drop "
                "--weights/--tokenizer and pass --model <store name> instead (install "
                "one with /install in the chat harness). To keep using a locally "
                "converted checkpoint, convert it with Tools/Converters (see "
                "Data/Models/README.md), then pass --weights / --tokenizer or set "
                "MILA_MODEL_PATH / MILA_TOKENIZER_PATH."
            )

    return weights_path, tokenizer_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(mila, family, weights, tokenizer, context_length, device_index=0, quantization=None):
    """
    Load a tokenizer and model pair for the given family, from loose artifact files.

    Quantization is a load-time choice here because the artifact is unquantized: "bf16",
    "fp8" or "fp4". Gemma defaults to FP4 -- a BF16 12B needs ~24 GB and OOMs on a 12 GB
    card -- and Llama to BF16.

    Prefer load_from_store() for an installed model: a published artifact is already
    quantized and only its store record says to what.
    """
    if family == "gemma":
        return (
            mila.BpeTokenizer.load_gemma(str(tokenizer)),
            mila.GemmaModel.from_pretrained(
                str(weights), context_length, device_index, quantization or "fp4"),
        )

    return (
        mila.BpeTokenizer.load_llama32(str(tokenizer)),
        mila.LlamaModel.from_pretrained(
            str(weights), context_length, device_index, quantization or "bf16"),
    )


def load_from_store(mila, name, context_length, device_index=0):
    """
    Load an installed model by store name -- as the artifact itself is.

    Returns (tokenizer, model, record). The record carries the architecture, the variant
    and whether the model is instruction-tuned, so a caller needs to know nothing about
    the model in advance. Nothing is downloaded: an uninstalled name raises.
    """
    record = mila.ModelStore().locate(name)

    if record is None:
        installed = ", ".join(model.name for model in mila.ModelStore().list())
        raise FileNotFoundError(
            f"No model named '{name}' is installed. Installed: {installed or 'nothing'}.\n"
            "Fetch one with mila.ModelStore().pull(name, mila.default_hub_owner()), "
            "or with /install in the chat harness. Loading never downloads."
        )

    session = mila.GemmaModel if record.architecture == "gemma" else mila.LlamaModel

    return (
        mila.BpeTokenizer.from_store(name),
        session.from_store(name, context_length, device_index),
        record,
    )
