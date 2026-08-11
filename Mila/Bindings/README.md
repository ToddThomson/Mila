# Bindings — Mila's Python projection

The `mila` pybind11 extension: a **runtime-adjacent** binding surface that projects the
Mila C++ runtime into Python. Module `Mila.Bindings`; built target `MilaPy` → `_mila`,
the private half of the `mila` package published to PyPI as `mila-llm`.

It is a peer of the runtime, **not an adaptor**. It is consumer-blind: everything it
exposes is model-intrinsic (`Tokenizer`, `LlamaSession` / `GemmaSession`: load, `generate`,
`generateStreaming`, `getConfig`) or store-intrinsic (`ModelStoreHandle`), and none of it
knows about chat, agents, or any wire protocol. That is why it lives here beside `Src`
rather than under `Adaptors/`.

It does carry HTTP *types* — `HttpResponseInfo`, `HttpFetchDelegate` — and that is not a
contradiction: the delegate exists so a build with no HTTP client at all can still pull,
by having Python move the bytes. The knowledge stays in the library (which URL, which
token, which digest); the binding projects a byte pipe, not a protocol.

Two consumers today, which is the point:

- **MIS** (`Mila/Adaptors/Inference/Server`) — the wire adaptor imports `mila` to serve the
  model over an OpenAI/Anthropic-compatible API.
- **Parity / converter tooling** (`Mila/Tools/Converters/...`) — the HuggingFace token-for-token
  parity harness imports `mila` directly; it is in fact `GemmaSession`'s primary consumer.

## Build

Gated by `MILA_ENABLE_PYTHON_BINDINGS` (default ON) and requires CUDA. The build stages the
extension into two neutral places and no consumer-specific one: the wheel source tree
(`Package/src/mila/`, so `pip install -e Mila/Bindings/Package` tracks every rebuild) and
`<build dir>/python/` (put on `sys.path` and `import mila` works). The MIS-specific copy was
removed — the server directory is first on `sys.path`, so it shadowed any installed
`mila-llm`.

## Design note

`Mila_py.cpp` binds against the std-only opaque handles declared in `Mila_py.Wrappers.ixx`
and never `import Mila;` — the latest VS2026 MSVC raises C2079 when `Mila` is imported into an
ordinary `.cpp` that also includes std headers. The Mila-touching bodies live in
`Mila_py.Wrappers.cpp`, kept unreachable to the binding TU so pybind never instantiates a Mila
template.
