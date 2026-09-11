# Contributor

Good-first-issue shaped: bounded, self-contained, and reachable by a strong developer without a
tour of the codebase. This is the **outbound** queue — these are what get mirrored *to* GitHub
Issues with a label when someone asks how to help.

An item here is not a commitment. Nothing in this file blocks a release; if it did, it would be in
[`BACKLOG.md`](../../BACKLOG.md) instead. Triage flow and categories are in
[README.md](README.md); the tag set is [Tags.md](Tags.md).

---

## Llama 3.2 1B/3B weight tying

`models` · `mila-src`

The aliasing plumbing shipped; what remains is `tie_word_embeddings_`, post-load aliasing, and the
`getMemoryStats` correction on `LlamaTransformer`. See `Specifications/WeightTying.md` §6.

## Llama-lineage CPU ops

`models` · `mila-src`

`RmsNormOp`, `SwigluOp`, `RopeOp`, `TokenEmbeddingOp` and `CrossEntropyOp` in
`OperationTraits.Cpu.ixx`. Demand-driven — their absence is zero-cost on the GPU path.

## `GemmaConfig::getRotaryDimForLayer()` is dead library code

`models` · `mila-src`

Its only callers are two assertions in `Gemma.Config.cpp:183,198`; the live path reaches the same
value through `rotaryDim()` → `getGlobalRotaryDim()` (`Gemma.Block.ixx:184`). Two names for one
concept, with the dead one reading as the live one.

Delete it and point the test at the live accessor. `Gemma.Config.ixx:536`

## A parity script cites a debug flag that no longer exists

`docs` · `observability`

`kGemmaDumpActivations` is gone from `Mila/Src`, but
`Gemma/gemma_4_BF16/hf_gemma_activation_dump.py:4` still tells the reader to diff its output
against it.

The replacement is `LanguageModel::observe` (`LanguageModel.ixx:130`) over `"*.tf_layer_*"`.
`GemmaModel::fingerprintPrefill` is **not** the substitute — it localizes a NaN rather than
comparing per-layer activations.

## A pre-flight that cannot answer says nothing at all

`adaptors`

`Chat::predictFootprint` (`Chat.ixx`) catches every exception and returns `nullopt`, so an
unreadable header shows as silence followed by a confusing failure at load.

One line at `verbose`. [[feedback_absent_output_is_evidence]]

## An `unknown` GPU FIT verdict prints no reason anywhere

`adaptors`

`verdictFor` distinguishes measured-and-too-big (`no`) from could-not-predict (`unknown`), but the
reason is discarded. It belongs at `/verbose all`, matching `reportFootprintBeforeLoad`.

The listing does not currently receive the detail level, so that has to be threaded through first.

## `temperature`/`top_k`/`top_p` have no command-line flags

`adaptors`

`/set` reaches them in a session and `session.json` at startup, so a `-p` one-shot cannot vary them
at all — the invocation most likely to want a fixed temperature.

`main.cpp:935` reads all three from settings already, so this is three flag producers, not a design.

## A test discards a `[[nodiscard]] GenerateStatus`, and warns on every build

`models` · `ci`

`QwenModel.Load.Cuda.cpp:189` calls `model->generate(...)` for its side effects inside a lambda,
producing C4834. The status is the only channel reporting why generation stopped, so a test ignoring
it cannot tell a completed run from an aborted one.

Assert it instead of casting it away.

## `GptModel.ixx:386` hardcodes `eos_token_ = 50256`

`models` · `mila-src`

It should come from tokenizer metadata.

## The Llama converter writes a metadata key the reader never parses

`models` · `docs`

It emits `norm_eps`; `parseMetadataJSON` extracts `norm_epsilon`, which is what Gemma and the packer
both emit. Harmless only because `LlamaModel::configFromMetadata` never reads the epsilon.

`Tools/Converters/Llama/convert_weights.py:188`

## `ToolCallParser::parse` routes any response containing `[` into the tool-call parser

`adaptors`

`Chat.ToolCallParser.ixx:63` uses `response.find( '[' )` where the class's own doc comment at `:35`
says "Leading `[`", and the nested `parseTagged` path at `:109` tests it correctly.

It degrades gracefully today, but any prose with a bracket enters the path, and a parse that ever
*succeeds* on prose would swallow the answer and emit a phantom tool call.

## `ModelSize` is dead

`adaptors`

Declared in `Chat.Config.ixx` with four values and read nowhere — the model's identity is its store
name, which is what replaced it. Left in place it invites the next family to add a fifth value that
nothing will ever read.

## Wrapped list items do not hang-indent

`adaptors`

A continuation line starts at the bullet's own indent rather than under the item text, so a wrapped
item reads as a new paragraph. `wordWrap` preserves a line's leading indent but has no notion of a
continuation indent. `Chat.RichText.ixx:99`

## `Chat.Json` duplicates the `nlohmann.json` module byte for byte

`adaptors` · `build`

`Mila/Adaptors/Chat/Src/Json.ixx` versus `Mila/Src/Utils/json.ixx`, both including the same header
from their global module fragment — and Chat imports one in `Chat.ixx` and the other in
`Chat.ModelCatalog.ixx`.

Drop `Json.ixx` from the target and import `nlohmann.json` everywhere.

## `isAbandoned()`'s 24-hour lock reclamation is untested

`distribution` · `ci`

It needs a file with a backdated write time. Make the threshold a constructor parameter so a test
can set it to zero — a better shape than backdating with `last_write_time()`.

## Two CUDA memory resources throw an allocation failure with no message

`mila-src`

`CudaManagedMemoryResource.ixx:85` builds a detailed error message and then throws a bare
`std::bad_alloc`, discarding it; `CudaPinnedMemoryResource.ixx:101` throws with no message at all.

`CudaDeviceMemoryResource` gets this right — align both on `CudaBadAlloc` so an OOM says which
device, which size, which resource.

## `actions/setup-python@v5` still declares Node 20, which GitHub has deprecated

`ci`

It warns on every clean-room run. Every other action in the tree is on `@v5` and clean; bump it and
re-check the rest — the deprecation applies by action version.
`.github/workflows/wheel-cleanroom.yml`

## The devel image's `mila-chat` wrapper shares its name with the binary it wraps

`build` · `docs`

`Docker/Dockerfile:94` installs `run-chat.sh` as `/usr/local/bin/mila-chat`; the binary is
`/build/mila-chat`. Its `cd /build` is redundant too — `executable_directory()` reads
`/proc/self/exe`, confirmed by the runtime image running `chat` from `-w /` and `-w /tmp`.

Drop the wrapper for a symlink, or keep it for the not-built message; either way drop the `cd` from
`Docker/run-chat.sh:24`.

## `Docker/README.md:69` credits ChatApp with a compiled-in `MODELS_DIR`

`docs`

It has none — the only `MODELS_DIR` in the tree is
`Mila/Profiling/ProfileModel/CMakeLists.txt:22`. Chat resolves models through `MILA_CACHE_DIR` and
its config through the working directory, which is why the published image can drop the bind mount.

The claim reads as a hard dependency on `/mila`.

## Delete the 16 `REVIEW:` markers whose disposition is already recorded

`mila-src`

No analysis left, only removal: the 12 in `CudaGqa.Dispatch.ixx` answered by that file's own banner
at `:36`, plus `CudaOps.h:30`, `Linear.cuh:83`, `Component.ixx:299` and
`CudaDeviceMemoryResource.ixx:139`.

## `Version::getMajor()`/`getMinor()`/`getPatch()` are non-const

`api` · `mila-src`

`Src/Version.ixx` — so the version-skew comparison needs a mutable copy.

## Document that CUDA's device 0 is not `nvidia-smi`'s

`docs` · `api`

`fromPretrained`'s default `DeviceId{ Cuda, 0 }` picks whichever card CUDA enumerates first, which
on a mixed-capacity machine can be the smaller one. A load sized for the larger card then aborts in
about two seconds with no diagnostic, and reads as a model defect rather than a device choice.

A note wherever the default device is documented. [[project_cuda_index_is_not_nvidia_smi_index]]
