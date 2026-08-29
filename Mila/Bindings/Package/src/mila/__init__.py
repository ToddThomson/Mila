"""
Mila -- a C++23/CUDA runtime for open large language models.

Read the loop, don't just call it: every forward pass is explicit, device and
precision are compile-time decisions, and there is no hidden execution engine.
This package is the Python projection of that runtime.

    import mila

    mila.initialize("warning")
    tokenizer = mila.BpeTokenizer.from_store("gemma-4-12b-it-fp4")
    model = mila.GemmaModel.from_store("gemma-4-12b-it-fp4", 4096)

    model.generate_streaming(tokenizer.encode(prompt), print)

A model is named, not pathed: from_store() reads the local store's record, which
is what knows the weights are already FP4. Install one first -- ModelStore().pull()
here, or `/install` in the chat harness -- because a load never downloads.

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
    "QwenModel",
    "StopController",
    # Chat protocol: the model's own grammar, projected from the runtime rather than
    # reimplemented here. A host renders a conversation; it does not write a template.
    "qwen_format_prompt",
    "qwen_parse_tool_call",
    "qwen_protocol_tokens",
    "gemma_format_prompt",
    "gemma_tool_declarations",
    "gemma_parse_tool_call",
    "gemma_format_tool_call",
    "gemma_format_tool_response",
    "gemma_extract_answer",
    "gemma_strip_control_tokens",
    "gemma_protocol_tokens",
    "cuda_library_directories",
    # Distribution: the local store, shared with Chat and the Inference Server.
    "ModelStore",
    "StoredModel",
    "StoreUsage",
    "RemovalReport",
    "HubModel",
    "HttpResponse",
    "default_hub_owner",
    "http_transport",
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
        QwenModel,
        RemovalReport,
        StopController,
        StoredModel,
        StoreUsage,
        default_hub_owner,
        gemma_extract_answer,
        gemma_format_prompt,
        gemma_format_tool_call,
        gemma_format_tool_response,
        gemma_parse_tool_call,
        gemma_protocol_tokens,
        gemma_strip_control_tokens,
        gemma_tool_declarations,
        initialize,
        qwen_format_prompt,
        qwen_parse_tool_call,
        qwen_protocol_tokens,
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


# ---------------------------------------------------------------------------
# HTTP transport
#
# The wheel is built with MILA_ENABLE_LIBCURL=OFF on BOTH platforms, so the
# extension has no HTTP client of its own and pull() would otherwise refuse --
# quoting a CMake flag at someone who ran `pip install`. On Linux that is forced
# (curl links the system OpenSSL, which manylinux does not whitelist); on Windows
# it is chosen, so that one code path serves both.
#
# Supplying a transport is cheap because the library kept all the knowledge: it
# builds the URLs, finds the token, sets the Range header, parses the manifest and
# verifies every digest. What follows only moves bytes.
# ---------------------------------------------------------------------------

#: Socket timeout, per read rather than per transfer -- a multi-gigabyte pull is
#: many reads, and a total-transfer deadline would abort a healthy slow download.
_HTTP_TIMEOUT_SECONDS = 60

#: Bytes per chunk handed to the store. The GIL is released around each write, so
#: this is the granularity at which a transfer yields the interpreter.
_HTTP_CHUNK_BYTES = 1024 * 1024

#: A failure body is a diagnostic, not a payload. Cap it so a server that answers
#: an error with a web page does not put megabytes into an exception message.
_HTTP_MAX_MESSAGE_BYTES = 8192

_http_opener = None


def _build_opener():
    """
    An opener that does NOT follow redirects.

    This is a security requirement rather than a preference, and it mirrors
    CURLOPT_FOLLOWLOCATION being off in the C++ transport. HuggingFace answers a
    weights request with a 302 to a DIFFERENT host (`us.aws.cdn.hf.co`), and
    urllib would carry the Authorization header across that hop. Reporting the
    Location and stopping lets the library decide whether the token travels.
    """
    import urllib.request

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        # Returning None leaves the 3xx unhandled, so urllib raises HTTPError and
        # the Location reaches the caller intact instead of being followed.
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    return urllib.request.build_opener(_NoRedirect)


def http_transport(url, headers, writer):
    """
    Move bytes for one request, using only the standard library.

    This is the default transport for ModelStore.pull() and list_hub_models(), and
    the reference for anyone writing their own (pass `transport=` to use it).

        url     the exact URL to fetch; never rewrite it
        headers a mapping to send verbatim -- the library has already put the
                token and any Range in here
        writer  call writer(chunk) for each chunk of a 2xx body; a False return
                means the store is stopping, so stop too

    Returns an HttpResponse. Report the status faithfully and do not interpret it:
    a Range answered 200 instead of 206 is the library's call to make, not this
    function's.
    """
    global _http_opener

    import urllib.error

    if _http_opener is None:
        _http_opener = _build_opener()

    import urllib.request

    request = urllib.request.Request(url, headers=dict(headers), method="GET")

    # The C++ build disables every compression codec, so the two transports agree
    # on identity encoding. It also matters for correctness: a re-encoded body
    # would not match the digest the manifest declares.
    if not request.has_header("Accept-encoding"):
        request.add_header("Accept-Encoding", "identity")

    try:
        with _http_opener.open(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            # urllib routes anything outside 200-299 to the error path, so only a
            # success body reaches the store -- which is what the C++ side does by
            # capturing non-2xx bodies as the message instead.
            location = response.headers.get("Location") or ""
            declared = response.headers.get("Content-Length")

            while True:
                chunk = response.read(_HTTP_CHUNK_BYTES)

                if not chunk:
                    break

                if not writer(chunk):
                    # The store refused the bytes. Not a transport failure: it knows
                    # why its own sink stopped, and saying otherwise would blame the
                    # network for a decision made above it.
                    return HttpResponse(
                        http_code=response.status,
                        location=location,
                        content_length=int(declared or 0),
                        transport_failed=False,
                        message="the store stopped the transfer")

            return HttpResponse(
                http_code=response.status,
                location=location,
                content_length=int(declared or 0),
                transport_failed=False,
                message="")

    except urllib.error.HTTPError as error:
        # Every non-2xx arrives here, including the 3xx this opener declines to
        # follow. The body is the server's explanation, so it becomes the message.
        try:
            detail = error.read(_HTTP_MAX_MESSAGE_BYTES).decode("utf-8", "replace")
        except Exception:
            detail = ""

        return HttpResponse(
            http_code=error.code,
            location=error.headers.get("Location") or "",
            content_length=0,
            transport_failed=False,
            message=detail)

    except Exception as error:
        # URLError, socket timeouts, TLS failures. transport_failed distinguishes
        # "could not ask" from "asked and was refused", which the library needs in
        # order to decide whether a retry could possibly help.
        return HttpResponse(
            http_code=0,
            location="",
            content_length=0,
            transport_failed=True,
            message=f"{type(error).__name__}: {error}")


_ExtensionModelStore = ModelStore


class ModelStore(_ExtensionModelStore):
    """
    The local model store, with a working default transport.

    Identical to the extension type except that pull() and list_hub_models() fall
    back to http_transport instead of to a build that has no HTTP client. Pass
    transport= to override, or transport=False to get the extension's own default
    back.
    """

    def pull(self, name, owner, transport=None):
        return _ExtensionModelStore.pull(
            self, name, owner,
            http_transport if transport is None else (transport or None))

    def list_hub_models(self, owner, transport=None):
        return _ExtensionModelStore.list_hub_models(
            self, owner,
            http_transport if transport is None else (transport or None))
