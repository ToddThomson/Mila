# Bindings

Mila from other languages. Each language has a folder:

| Folder | What it is | Status |
|---|---|---|
| [`Python/`](Python/) | The `mila` extension, published to PyPI as `mila-llm` | Shipped |
| [`DotNet/`](DotNet/README.md) | A C# library over the C interface | Experimental; runs against a mock (`Specifications/DotNetBinding.md`) |
| [`Native/`](Native/mila.h) | `mila.h`, the C interface the C# library calls | Experimental |

Every binding is **consumer-blind**, and that is why the bindings live in the library rather than in an app's project:
everything one exposes is about a model or the model store, and none of it knows about Chat, the inference server or a
wire protocol. A binding projects sessions, never components.

## Python

Built target `MilaPy`, which makes `_mila`, the private half of the `mila` package. `Python/Package/` is the rest of the
package: `__init__.py`, which registers the CUDA library folders before the extension loads, and what the wheel is built
from. The module is `Mila.Bindings`.

It exposes models (`Tokenizer`; `LlamaSession`, `GemmaSession` and `QwenSession`: load, `generate`, `generateStreaming`,
`getConfig`), the store (`ModelStoreHandle`), what runs on a device, and each chat template's protocol.

It does carry HTTP *types* -- `HttpResponseInfo`, `HttpFetchDelegate` -- and that is not a contradiction: the delegate
exists so a build with no HTTP client at all can still pull, by having Python move the bytes. The knowledge stays in the
library (which URL, which token, which digest); the binding projects a byte pipe, not a protocol.

Two consumers today:

- **The inference server** (`Apps/Server`) imports `mila` to serve a model over OpenAI- and Anthropic-compatible APIs.
- **Parity and converter tooling** (`Lib/Tools/Converters/`): the HuggingFace token-for-token parity harness imports
  `mila` directly, and is `GemmaSession`'s primary consumer.

### Build

Gated by `MILA_ENABLE_PYTHON_BINDINGS` (on for a top-level build) and requires CUDA. The build stages the extension into
two neutral places: the wheel source tree (`Python/Package/src/mila/`, so `pip install -e Lib/Bindings/Python/Package`
tracks every rebuild) and `<build dir>/python/` (put it on `sys.path` and `import mila` works).

### Design note

`Mila_py.cpp` binds against the std-only opaque handles declared in `Mila_py.Wrappers.ixx` and never does `import Mila;`.
Its stated reason is that MSVC raises C2079 when `Mila` is imported into an ordinary `.cpp` that also includes standard
headers. The bodies that touch Mila live in `Mila_py.Wrappers.cpp`, unreachable from the binding's translation unit, so
pybind11 never instantiates a Mila template.
