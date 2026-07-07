# Bindings — Mila's Python projection

The `mila` pybind11 extension: a **runtime-adjacent** binding surface that projects the
Mila C++ runtime into Python. Module `Mila.Bindings`; built target `MilaPy` → `mila.pyd`.

It is a peer of the runtime, **not an adaptor**. It is consumer-blind — it exposes only
model-intrinsic surface (`Tokenizer`, `LlamaSession` / `GemmaSession`: load, `generate`,
`generateStreaming`, `getConfig`) and knows nothing about HTTP, chat, or any wire protocol.
That is why it lives here beside `Src` rather than under `Adaptors/`.

Two consumers today, which is the point:

- **MIS** (`Mila/Adaptors/Inference/Server`) — the wire adaptor imports `mila` to serve the
  model over an OpenAI/Anthropic-compatible API.
- **Parity / converter tooling** (`Mila/Tools/Converters/...`) — the HuggingFace token-for-token
  parity harness imports `mila` directly; it is in fact `GemmaSession`'s primary consumer.

## Build

Gated by `MILA_ENABLE_PYTHON_BINDINGS` (default ON) and requires CUDA. The build drops
`mila.pyd` into the MIS server directory as a convenience so `python main.py` imports it
without PYTHONPATH setup — the one consumer-specific reach the binding still makes; a neutral
output location that both consumers pull from is a recorded follow-up.

## Design note

`Mila_py.cpp` binds against the std-only opaque handles declared in `Mila_py.Wrappers.ixx`
and never `import Mila;` — the latest VS2026 MSVC raises C2079 when `Mila` is imported into an
ordinary `.cpp` that also includes std headers. The Mila-touching bodies live in
`Mila_py.Wrappers.cpp`, kept unreachable to the binding TU so pybind never instantiates a Mila
template.
