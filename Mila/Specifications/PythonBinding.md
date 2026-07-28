# Python Binding and Samples

Design notes for `mila.pyd` (the pybind11 projection, module `mila`) as a **consumable product**
rather than an internal detail of the Mila Inference Server.

Companion to [MilaProductFamily.md](MilaProductFamily.md). MIS is the Python *wire* adaptor; this
document covers the binding itself and the samples that make it approachable to a Python-first
audience.

---

## Why

Most people who run local LLMs work in Python. Mila's binding already exists and already drives a
real workload — MIS serves Gemma 4 to Claude Code and Codex through it — but it has exactly one
consumer, inside this repository, and no user-facing entry point. Nobody outside the project has ever
seen it as an API.

The gap is not capability. It is that reaching Mila from Python currently requires building the
library, locating a built artifact, converting weights, and knowing where everything landed.

---

## Current surface

Complete as of `0.20.0-beta.2+16`. Source: `Mila/Bindings/Mila_py.cpp`.

| Symbol | Members |
|---|---|
| `mila.initialize` | `log_level` = `trace \| info \| warning \| error` |
| `mila.BpeTokenizer` | `load_llama32`, `load_gemma`, `encode`, `decode`, `token_to_string`, `is_valid_token`, `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id` |
| `mila.LlamaModel` | `from_pretrained(path, context_length, device_index=0, quantize_fp8=False)`, `generate`, `generate_streaming`, `get_config`, `__repr__` |
| `mila.GemmaModel` | `from_pretrained(path, context_length, device_index=0)`, `generate`, `generate_streaming`, `get_config`, `__repr__` |
| `mila.StopController` | `request_stop`, `stop_requested` |

Two properties worth stating because they make a real sample possible: **the GIL is released around
generation** (`py::gil_scoped_release`), so streaming callbacks and a Ctrl-C handler both work; and
**`GemmaModel` hardcodes FP4** (`.withFP4Quantization()`, with the 12 GB reasoning recorded at the
call site), so the 12B flagship loads on a consumer card rather than OOM-ing at BF16.

**Not exposed:** `GptModel`. This matters more than it looks — see *Weights* below.

---

## Defects and gaps

Found while scoping, 2026-07-28. Tracked in BACKLOG.

- ~~**`quantize_fp8` is accepted and silently ignored.**~~ **Fixed 2026-07-28.** The argument reached
  a lambda body of `(void)quantize_fp8;`, so a caller asking for FP8 received BF16 with no error and
  no warning. It now applies `WeightQuantization::FP8` to the Linear weights. Weights only:
  `withFP8Quantization()` would also request an FP8 KV cache, which `LlamaModel::fromPretrained` does
  not implement.
- ~~**The module docstring is stale**~~ — **Fixed 2026-07-28.** It said *"Mila inference bindings —
  Llama 3.2 3B Instruct on CUDA BF16"* while Gemma was bound and is the flagship, and it is the first
  thing `help(mila)` prints. It now names both models and states the GIL-release property.
- ~~**`mila.pyd` is copied only into `Mila/Adaptors/Inference/Server/`.**~~ **Fixed 2026-07-28.**
  `MilaPy` publishes to `<build dir>/python/` — a directory holding nothing but the extension, so a
  consumer can put it on `sys.path` without dragging that consumer's sources along. Exported as the
  `MilaPy_PYTHON_DIR` cache variable. The MIS copy is retained, since its documented run instructions
  depend on it; it is simply no longer the only place the extension lands.
- **No precision control for Gemma.** Reasonable today (FP4 is the only configuration that fits the
  target card) but it means the Python API cannot express a choice the C++ API can. Still open —
  adding the parameter is a feature addition the v0.20 freeze excludes.

---

## The "one click" goal

Target: a Python user goes from a clean checkout to streaming tokens **in one command, with no
background setup**. Broken into what is actually blocking, because the tiers have very different
costs.

### Tier 1 — one command, given a built `mila.pyd`

**Delivered 2026-07-28.** With the neutral output location in place, `python Mila/Samples/Python/chat.py`
is the whole command: the sample finds the extension under `out/build/*/python/` (or `MILA_PYD_DIR`),
finds the weights under `Data/Models/` (or `MILA_MODEL_PATH`), and streams. Two failure modes get a
sentence rather than a stack trace — an extension built for a different Python (the ABI tag is
matched, and both tags are named), and missing weights (which point at the converter).

A third was not a diagnostic but a genuine blocker, found on first run: **`PATH` is not searched when
Windows resolves an extension module's DLL dependencies** (Python 3.8 tightened this to system
directories, the extension's own directory, and `os.add_dll_directory`). The binding links
`cublasLt` / `curand` / the CUDA runtime dynamically, so `import mila` fails with a bare "DLL load
failed" on a machine with a correct CUDA install on `PATH`. The sample registers the toolkit's
`bin\x64` and `bin` before importing.

This is not sample-local. MIS handled it inline at the top of `main.py`, which covered the server and
nothing else: importing `model_worker` or any route module directly still failed, as did the README's
own `python -c "import mila"` verification step, and the inline version raised `FileNotFoundError` on
a stale `CUDA_PATH`. Hoisted 2026-07-28 into `Server/cuda_runtime.py`, imported ahead of `mila` in all
five modules that touch the binding. **Any future consumer of the binding needs the same three lines**
— which is an argument for Tier 3 (a wheel) doing it once in a package `__init__`, rather than each
consumer rediscovering it.

### Tier 2 — zero-auth weights, fetched on first run

**Newly possible in Python, and this is the key insight.** The equivalent C++ quick-start was
deferred to vNext because fetching weights over HTTPS meant a runtime addition and a new HTTP
dependency. Neither applies here: `urllib.request` is in the Python standard library, and a sample is
not the runtime. A first-run download with a progress bar is ordinary sample code, adds no
dependency, and does not touch `Mila/Src`.

What it needs is a **hosted, pre-converted artifact** — the `.bin` weights plus the tokenizer binary
— so the user never runs the converter. That removes the real cliff, which was never the download; it
was requiring Python, PyTorch and a HuggingFace token just to produce a `.bin`.

**Licensing is not a blocker: Gemma 4 12B is Apache 2.0.** This is a change from earlier Gemma
releases, which shipped under the bespoke Gemma Terms of Use — do not reason from those. Apache 2.0
permits redistribution, modification and commercial use outright, and the obligations are the
standard four: include the licence, include the `NOTICE` file, retain attribution notices, and state
that the files were modified. A converted FP4 `.bin` is a modified work, so the modification notice
applies; nothing else in the earlier Gemma terms — prohibited-use pass-through, terms-acceptance
chains — carries over. Attribution belongs in the root `NOTICE.md`, consistent with the standing rule
that everything Mila did not write is recorded there.

Two things Apache 2.0 does **not** grant, worth stating so no one assumes otherwise: rights to
Google's trademarks, names or logos, and any warranty — the weights are as-is.

This resolves what first looked like a trap. The model with the least redistribution friction was
assumed to be GPT-2 — which is why the deferred C++ quick-start targeted it — but `GptModel` is
**not bound**, and binding it would be a feature addition barred by the v0.20 freeze. Gemma 4 under
Apache 2.0 is at least as unencumbered, so Tier 2 needs neither GPT-2 nor a new binding and is
**freeze-compatible end to end**.

It also inverts the earlier artifact reasoning. Llama 3.2 carries the Llama Community License with
its naming and threshold conditions; Gemma 4 carries none of that. **Gemma 4 12B FP4 is therefore
both the flagship and the licensing-simplest option**, and the only argument against leading with it
is download size.

A smaller model improves first-run experience considerably: Llama 3.2 1B is a far better "first
token in under a minute" story than a multi-gigabyte 12B download.

### Tier 3 — no build at all

A published wheel. This is the only thing that makes Mila reachable by someone who will never open
Visual Studio, and it is the tier that actually serves the stated audience.

**Started 2026-07-28 by explicit direction** — the tier the audience actually needs, so the earlier
"post-v0.20, scope it on its own" deferral was overridden rather than forgotten.

**Decided:**

- **Distribution name `mila-llm`; import name stays `mila`.** The two are independent. Bare `mila` on
  PyPI is a derelict 0.0.1 from 2022 — the PyPA sample project published verbatim, placeholder
  metadata intact, 19 downloads a month. A PEP 541 claim is being filed against it; `mila-dnn` was
  considered and rejected, since accuracy beats differentiation bought with dated vocabulary, and
  assistant-mediated discovery favours the domain word. Note that Mila Quebec already publishes
  `milatools`, so the prefix is shared.
- **CUDA arrives as `nvidia-cublas` + `nvidia-curand` dependencies**, not an assumed Toolkit — that is
  what makes `pip install mila-llm` work on a machine with no CUDA installed, which is the whole point
  of a wheel. The `-cu13` spellings are deprecated stubs; the live packages are unsuffixed and publish
  **win_amd64** wheels, so this works on Windows and not only Linux. An installed Toolkit remains the
  fallback.
- **The extension became `mila._mila`, behind a `mila/__init__.py`.** Not cosmetic: Windows does not
  search `PATH` for an extension's DLL dependencies, so *something must run before the extension
  loads*. A bare top-level extension has no such place. The package `__init__` is that place, which
  single-sources the CUDA registration for every consumer — sample, MIS, and wheel alike.
- **Wheels are per CPython minor per platform.** cp313-win_amd64 today; each additional interpreter and
  platform is another build, and that matrix is what publishing actually costs.

Backend naming was considered and dropped: `mila-cuda` would pigeonhole a runtime that reserves
`DeviceType::Rocm` and `::Metal`. The multi-backend answer is JAX's, not cupy's — a neutral
distribution with additive backend extras (`mila-llm[rocm]`), so no user's pin ever breaks.

---

## Samples

Shipped 2026-07-28 at `Mila/Samples/Python/`. No pip dependencies — standard library only. The
absence of a `requirements.txt` is itself part of the message.

- **`chat.py`** — the flagship. Load Gemma, tokenize, stream tokens to stdout, interrupt through
  `StopController`. This is the Chat adaptor's job in roughly a hundred readable lines, which is the
  argument to this audience: here is the whole loop, in your language, with nothing hidden. Gemma-only
  by choice: the instruct template and the channel filter *are* the content, and a family switch
  would bury them in branching. Ctrl-C installs a `SIGINT` handler that calls `request_stop()` rather
  than letting `KeyboardInterrupt` raise through C++ with a forward pass in flight.
- **`generate.py`** — a smaller script exercising the sampling knobs (`temperature`, `top_k`,
  `top_p`) and a tokenizer round-trip. This is what enthusiasts actually poke at. Covers both
  families, and `--fp8` is the user-facing exercise of the repaired defect.
- **`common.py`** — a fourth file the original scope did not name: locating the extension and the
  weights, plus the two diagnostics above. It exists so the other two open with the part worth
  reading rather than thirty lines of `sys.path` bootstrap, duplicated.
- **`README.md`** — how to run, and an honest statement of what the binding does and does not
  expose. Under the reference-implementation positioning, the limits are part of the documentation,
  not an omission from it.

---

## Freeze boundary (v0.20)

In bounds — sample code, documentation, packaging, and defect repair:

- ~~the samples and their README~~ — landed 2026-07-28
- ~~the neutral `mila.pyd` output location~~ — landed 2026-07-28
- ~~the `quantize_fp8` fix~~ — landed 2026-07-28, honoured rather than rejected
- ~~the stale module docstring~~ — landed 2026-07-28
- a first-run download written in sample code against the standard library — **still open**, and
  blocked only on hosting the converted artifact (open decision 1)

Out of bounds — feature additions, deferred to vNext:

- binding `GptModel`
- adding a precision parameter to `GemmaModel.from_pretrained`
- a published wheel

---

## Open decisions

1. **Publish converted Gemma 4 12B FP4 weights to a HuggingFace repository.** Required for Tier 2.
   Apache 2.0 settles the licensing question; what remains is mechanical — licence, `NOTICE`,
   modification statement — and a product call on first-run experience, since a multi-gigabyte
   download is a slow first impression. Worth checking whether a smaller Gemma 4 variant exists that
   Mila could validate, which would give the good first run without the Llama licence conditions.
2. **Whether the Python sample is a v0.20 barrier lever.** It is the same class as the Docker image
   and the CPU-only path — work that lets an audience reach Mila at all. Promoting it from `## Future`
   into Production Hardening changes what v0.20 claims, so it is a deliberate call.
3. **Wheel distribution** (Tier 3), and whether it belongs with the Python work or with packaging.
