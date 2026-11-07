# Device Assumptions

Where Mila assumes the device is CUDA, swept on 2026-11-07 at `0.21.0-dev+125`, before Chat runs on the Mac
(`Metal/MetalBackend.md` Phase 8). Each site is sorted as **correct** (a fact about CUDA, or a CUDA-only feature
gated where it should be), **wrong** (it assumes the device is CUDA), or **stale** (text only). The decisions it
works under are `Deployment.md` 12.8 and `LayerSplit.md` 12.2, agreed the same day; the open one it informs is where
device and precision are chosen (section 9).

---

## 1. Method

Text searches over `Lib/Src`, `Lib/Store`, `Apps/` and `Lib/Bindings`, then a reading of each hit that is code:

- `cuda`, case-blind: 427 hits in 115 files of `Lib/Src` outside `Devices/Cuda`, 65 in 25 files of `Apps/`,
  141 in 23 of `Lib/Bindings`. Most are comments, and those under `Devices/Metal` cite CUDA's design.
- `DeviceType::Cuda`, `Device::Cuda(`, `CudaDevice`, `MILA_HAS_CUDA`, CUDA imports and runtime calls, outside
  `Devices/Cuda` and `Devices/Metal`: the code this notebook sorts.
- `device_index`, `"device"`, `--device`, `VRAM`, `ordinal` in the apps and bindings.

Not swept: `Lib/Tests` (a test file per device, `*.Cuda.cpp`, by design), `Lib/Samples`, `Lib/Tools`, `Web/` and
the user documents. A search finds what names CUDA; a site that assumes CUDA without naming it is found only by
running on another device, which is what Phase 8 does.

## 2. Choosing a device

| Site | Kind | What it does | Fix |
|---|---|---|---|
| `DeviceId()`, `Compute/Devices/DeviceId.ixx:44` | wrong | A default-constructed id is CUDA 0 | No default device in the type; which device to use when none is named is the planner's (`Deployment.md` 12.8) |
| `DeploymentRequest::withDevice`, `Deployment/Planning/DeploymentRequest.ixx:103` | wrong, consistent today | One device; unset is "the family's device type at ordinal 0" | Becomes the allowed devices (12.8) |
| `getBestDeviceId`, `Devices/Cuda/Helpers/CudaHelpers.ixx:142`; `getBestDevice`, `Registry/DeviceRegistryHelpers.ixx:81` | wrong | Ranks CUDA devices by multiprocessors, threads and clock, or memory and bus width; every other type takes the first registered. Only `Lib/Tests/.../DeviceRegistryHelpers.cpp` reaches it. From 2025-05 (`31dc07c5`), before the planner | Delete both and their test: choosing a device is the planner's |
| `Device::getDeviceName()`, `Devices/Device.ixx:61` | wrong (missing) | Answers a coordinate, `CUDA:0`. `Device` has no hardware name and no stable identity, so Chat and the binding cast to `CudaDevice` for the name and PCI address | `Device` gains the hardware's name and a stable identity (section 9, question 2) |
| `Agent`, `Agents/Agent.ixx:1610` | wrong | The application's agent is CUDA and BF16 | Section 9, question 1 |
| `planDeployment`, `fittingWeightQuantizations`, `Deployment/Planning/InstalledModelPlanner.ixx:56`, `:87` | wrong | Precision defaults to BF16 on every device; Metal runs FP16 (`MetalBackend.md` 3.2) | The precision follows the device (section 9, question 1) |
| `requireDevice`, `GemmaModel.ixx:695`, `LlamaModel.ixx:549`; `validateDeployment`, `QwenModel.ixx:764` | correct, finding | A request whose device type is not the model's is `std::invalid_argument`, not a refusal an application can word | A device type the build lacks is refused before any model type is chosen, naming the devices it has; these checks then cannot fire |
| `createExecutionContext`, `ExecutionContext/ExecutionContextFactory.ixx:18` | correct | Already turns a run-time `DeviceId` into a typed context for each backend built | The precedent for choosing a device at load |
| `DeviceType`, `Devices/DeviceType.ixx:22-23` | stale | Metal and ROCm marked `FUTURE` | Metal's mark goes; ROCm's stays until a backend exists |
| `Devices/Rocm/` | wrong | A Vulkan placeholder; a ROCm build cannot compile | Filed, `Issues/Untriaged.md` |

## 3. Capabilities written as "CUDA or not"

Written when the only other device was the CPU, so "not CUDA" meant "the host".

**Features only CUDA has.** `Linear::decode` (`Linear.ixx:201`), `GatedMLP::decodeTokens` (`:172`),
`Router::decodeTokens` (`:124`), Gemma's dense and routed feed-forward `decodeTokens`, `kDraftModelRuns`
(`Gemma.ixx:136`), the cache views a draft model reads (`Gemma.ixx:639`, `:652`, `Gemma.Block.ixx:448`) and
attention's multi-token paths (`GroupedQueryAttention.ixx:364-400`). **Correct today:** Metal has no multi-token
decode, filed in `Untriaged.md` ("Speculative decoding runs on CUDA only"). The shape is a device comparison where
Phase 4 put the weight dispatch on `OperationSupported`; each gate moves to that when Metal gains the operation.

**Where host memory sits.** The embedding table's residency (`Qwen.ixx:137`), `TokenEmbedding`'s host-resident
assert (`:126`), host staging for Gemma's layer scalar (`Gemma.Block.ixx:616-675`, `:866`; `Gemma.DraftBlock.ixx`
the same), pinned staging when a parameter loads from an archive (`Component.ixx:485`), and pinned streaming of
weights at load (`Gemma.ixx:843`, `Llama.ixx:640`, `Qwen.ixx:648`, `GptTransformer.ixx:563`, and
`WeightsReader.ixx:396`). On Metal each takes the host branch, which is right for unified memory, and the comments
give the CPU as the reason. **Correct outcome, wrong reason:** a ROCm device would also take the host branch,
which is wrong for a card with memory of its own. Fix: each asks one device-type trait naming the property --
whether the device has memory of its own fed from pinned host memory -- in place of `== Cuda`.

**One allocation per block's weights,** `CompositeComponent.ixx:1046`: CUDA only. It recovers CUDA's 2 MiB
rounding (`MemoryFootprint.md` 11.8.1); whether Metal rounds is not measured, so whether Metal wants it is
unknown.

## 4. Memory types

| Site | Kind | What it does | Fix |
|---|---|---|---|
| `DeviceTensor`, `PinnedTensor`, `UniversalTensor`, `Tensors/Tensor.ixx:1022` | wrong | `DeviceTensor` names a CUDA tensor | Name them for CUDA, or remove them; callers not yet counted |
| `get_alignment`, `Tensors/TensorBuffer.ixx:70` | not checked | A CUDA buffer aligns to a warp's elements; every other one, Metal's included, to the CPU's SIMD width | Read what Metal's buffers need |
| `MemoryResourceTraits`, `Compute/Memory/MemoryResourceTraits.ixx:30-34` | stale | Carries OpenCL and Vulkan flags for backends that do not exist | Trim when touched |
| `DataLoader`, `TokenSequenceLoader`, `AdamW` | correct | Pinned staging and an optimizer for training, which only CUDA does (`DeviceTypeTraits::supports_training`) | None |

The model files import `Compute.DeviceTypeTraits.Cuda` under `MILA_HAS_CUDA` and no Metal counterpart
(`GemmaModel.ixx:50`, `LlamaModel.ixx:47`, `QwenModel.ixx:61`), and still build on Metal. Why they compile there
has not been read; it is noted, not sorted.

## 5. Library documentation

Doxygen that names CUDA as the only device: `@tparam TDeviceType Device type (DeviceType::Cuda)` on `Residual`,
`Lpe`, `Softmax`, `Gelu`, `TokenEmbedding` and `OptimizerBase`, and examples in `Network`, `NetworkFactory`,
`ComponentFactory`, `Gelu` and `Softmax`. **Stale.** Each says the device type is a template argument, where it
is touched.

## 6. Applications

**Chat** (`Apps/Chat/Src`). **Wrong, every site, and the decisions cover them:**

- The device is an `int` read as a CUDA ordinal: `Chat.SettingsResolution.ixx:532`, `Chat.Config.ixx:81-92`,
  `Chat.FootprintBudget.ixx:43`, `main.cpp:292-297`, and its readers in `Chat.ConsoleFrontEnd.ixx:889`, `:910`,
  `:1342`, `:1461`.
- The request names CUDA: `Chat.Footprint.ixx:64`, `Chat.ModelCatalog.ixx:449`; the fit column plans CUDA,
  `fittingWeightQuantizations<DeviceType::Cuda>`, `Chat.ModelCatalog.ixx:459`.
- The device is asked for as CUDA: memory, `Chat.Footprint.ixx:32`; the card's name by a cast to `CudaDevice`,
  `:47`.
- What the person reads: "CUDA ordinal" (`main.cpp:367`), "GPU Fit" and "VRAM" (`Chat.ModelCatalog.ixx:412`,
  `:534`), "VRAM" (`Chat.ConsoleFrontEnd.ixx:1346`, `:1705-1710`), "this GPU" in each refusal
  (`Chat.Session.ixx:1608-1624`). A Mac has unified memory, so "VRAM" is untrue there.

**Mila Desktop's session.** The device is an `int` sent as CUDA (`Desktop.SessionProcess.ixx:1162`,
`main.cpp:116`, `:151`): **wrong**. Its package copies CUDA's libraries beside the session
(`Session/CMakeLists.txt:67-84`): **correct** for the Windows package, and a Mac package carries its own list.

**The build.** Chat and Desktop's session build only with CUDA (`CMakeLists.txt:365`): **wrong**.

**`mila-store`.** The fit column asks CUDA alone (`Lib/Store/Store.ixx:216`, `:282`), so on a Mac it is empty:
**wrong**.

**The server** (`Apps/Server`). `device_index`, "CUDA device ordinal" (`config.py:43`), `cuda_devices()` in the
listing and the worker (`cli.py:22-68`, `model_worker.py:149-285`), `MILA_DEVICE_INDEX` in the README:
**wrong**; `LayerSplit.md` 8 already replaces it with `MILA_DEVICES`. The `NVIDIA CUDA :: 13` classifier is
**correct** while the server's binding is CUDA only.

## 7. Bindings

**Python** (`Lib/Bindings/Python`).

- It refuses to configure without CUDA (`CMakeLists.txt:8`): **wrong** against `ROADMAP.md` *Mila on Apple
  Silicon*, which ships a macOS wheel.
- Per-family sessions typed CUDA and BF16 (`Wrappers/Mila_py.Wrappers.cpp:25-27`), a request naming CUDA (`:354`),
  refusals saying "CUDA device" (`:399-421`), the fit query (`Mila_py.Deployment.cpp:63-71`), "CUDA device index"
  in every docstring: **wrong**. The sessions are the ones `Direction.md` 4.2 rule 5 retires for `mila.Agent`, so
  most of this goes with them.
- `cuda_devices()` and `CudaDeviceInfo` (`Mila_py.Runtime.ixx`, `.cpp`, `Mila_py.cpp:488-512`): **wrong**; the
  listing is of the devices the build has, whatever their type.
- The package loading NVIDIA's libraries (`Package/src/mila/__init__.py`) and their wheels as dependencies
  (`pyproject.toml:51-65`): **correct** for the CUDA wheel; a macOS wheel has neither. The description, "on CUDA",
  is stale for the macOS wheel.

**.NET and the C ABI** (draft, `DotNetBinding.md`). `mila_cuda_devices_json` (`Native/mila.h:68`), `CudaDevice`,
`LoadOptions.DeviceIndex`: **wrong**, the same listing and the same index. It is experimental, so it changes with
the binding.

## 8. Documents

- `Direction.md` 1.2, "it goes deep on NVIDIA rather than wide", and the layer diagram's bottom row, "CUDA / CPU":
  stale.
- `MilaDesktop.md`, "macOS once a Metal backend exists" (6.1) and "once a Metal backend exists" (section 9, decision 6): stale; one exists.

## 9. What the decisions cover, and what they leave

**Covered** by the allowed devices (`Deployment.md` 12.8) and a handle that chooses device and precision at load:
every application site in section 6 but the packaging, the binding's request and refusal text, and the precision
default in section 2. An application then names no device type, and on a Mac it gets Metal because Metal is what
the build has.

**Not covered**, each its own change:

- `DeviceId`'s default and `getBestDevice`, section 2.
- A hardware name and a stable identity on `Device`, section 2, which a listing and a stored setting both need.
- The tensor aliases and the alignment, section 4.
- The host-memory trait, section 3: no defect on Metal, one on any card with memory of its own.
- The Python binding's CUDA gate and a macOS wheel's packaging, Phase 8.

**Open, Todd's:**

1. **Where device and precision are chosen.** Amend `ModelHandle.md` 4 and `Direction.md` 4.1 so they are
   chosen at load, in the handle, with the precision the device's (`MetalBackend.md` 3.3), or keep them a
   program's compile-time choice. Section 2's `createExecutionContext` is the precedent one level down.
2. **The identity a stored setting names.** A CUDA index moves with `CUDA_VISIBLE_DEVICES` and is not
   nvidia-smi's; a UUID does not move. A Mac has one device.
3. **`"device": 1` in settings already on disk.** Read as before, or refused with the new form named.

**Question 1 decided the same day (Todd):** chosen at load, with no device or precision parameter above the
network, and `Agent` and the handle plain classes (`ModelHandle.md` 10.7). Neither parameter is in a public
member of `LanguageModel` or `Model`; Todd's question was whether the agent needs a precision at all.

**Question 2 decided the same day (Todd):** a device is named by its index or by a stable identity, and a
stored setting by the identity (`Deployment.md` 12.8). CUDA's is the UUID in `cudaDeviceProp`, present in the
13.1 to 13.5 headers on the development PC; `CudaDeviceProps` keeps the name and PCI address today, not it.

**Question 3 decided the same day (Todd):** a bare number is refused, naming the new form -- Chat's
`"device"` and `--device`, and the server's `MILA_DEVICE_INDEX`; the binding's `device_index=` becomes
`devices=` (`LayerSplit.md` 8, `ChatConfiguration.md`).

**Fixes agreed the same day (Todd), to build:**

1. `DeviceId` loses its default constructor, so every id names its device type and the build finds each site
   that relied on CUDA 0.
2. `getBestDevice`, `getBestDeviceId` and `findCudaDevice` are deleted with their test: a second place that
   chooses a device. What they read is kept -- each `Device` reports its performance facts raw (on CUDA the
   multiprocessors, clock, memory and bus width, and bandwidth where reported), the planner's `DeviceReading`
   carries them, and the objective does not use them until a measured rule does (`Deployment.md` 12.9). Todd:
   the planner should know the fastest device, without using it yet.
3. `Device` gains the hardware's name, a stable identity and the performance facts, one change: CUDA adds the
   UUID `CudaDeviceProps` does not keep; Metal names its GPU from what `MetalDevice` reads and is identified as
   the one GPU; the CPU is `cpu` with no name or facts yet. `getDeviceName()` stays the coordinate.
   `DeviceId::parse` accepts a stable identity beside `type:index`; its written form is confirmed from NVIDIA's
   documentation in the code proposal. Chat's `CudaDevice` cast and the bindings' CUDA listings move in the
   application and binding changes.
4. `DeviceTensor<TDeviceType, T>` becomes a tensor in that device's own memory, through
   `DeviceTypeTraits<TDeviceType>::memory_resource`; `PinnedTensor` becomes `CudaPinnedTensor` and
   `UniversalTensor` `CudaManagedTensor`. 262 uses in 35 files, the CUDA tests and `ExportArtifact`; mechanical,
   and it may share a change with the `TPrecision` rename.
5. Section 3's host-memory sites were a reading short: `DeviceTypeTraits` already declares
   `host_staging_memory_resource` for CUDA (pinned), the CPU and Metal (its shared memory), and `Component.ixx:171`
   and `TokenEmbedding.ixx:159` use it. The staging sites -- pinned streaming at load in the four transformers,
   `Component.ixx:485`, Gemma's layer-scalar staging -- use that trait in place of `== Cuda`. One new trait says
   whether the device's memory is separate from the host's (CUDA yes; Metal and the CPU no), and the embedding
   table's residency and `TokenEmbedding`'s assert ask it. On Metal the layer scalar then stages in shared memory
   rather than CPU memory, so Metal's Gemma tests run on the change.

**Change 1 built, 2026-11-07.** `Device` reports a hardware name, a stable identity and its speed facts; CUDA's
identity is the UUID, read as `GPU-11770557-a821-1b71-35db-34e342bae260` for the RTX 4070 and
`GPU-9a81c7d1-9db2-16b3-c256-2f991ec2a22c` for the RTX 5060 Ti, both as `nvidia-smi -L` prints them, where CUDA numbers
the 4070 device 0 and nvidia-smi numbers it 1. `DeviceId::parse` stays `type:index`: an identity becomes an index only
against the running process, so `findDeviceByStableIdentity` resolves it through the registry, taking the whole
identity or a prefix one device has, case ignored, as `CUDA_VISIBLE_DEVICES` does (NVIDIA's CUDA Programming Guide,
*CUDA Environment Variables*), and refusing a prefix two devices share, naming both. `CpuDeviceRegistrar` and
`CudaDeviceRegistrar` moved to files of their own, the latter undoing an earlier MSVC internal-error workaround;
MSVC 18.6 compiled it. Windows `x64-claude-verify`: every target, no warnings; `MilaTests` 2508 pass, 12 skip, 4 fail,
the four filed ones (the partial 26B weights, three; the 12B drafter near-tie); the three other binaries pass. Forced
failures: two of eleven run, both caught (the UUID's grouping, its case); the run was stopped for its cost (each mutant
rebuilt about 530 targets, `Issues/Untriaged.md`), the four on the lookup's rules not run, and the five on one-line
pass-throughs judged not worth a rebuild each (Todd). The Mac (M1 Pro, Clang, `macos-clang-release`): builds;
`MilaTests` 1752 pass, 3 skip (two CUDA-only identity tests, a CUDA buffer test), 7 fail, all filed before this change
(six Llama FP8 HuggingFace-reference runs, `Issues/Untriaged.md`; the BF16 log-likelihood miss,
`Notebooks/MetalFirstModel.md` 2); Metal's device reports "metal" and its chip's name; the three other binaries pass.

**Change 2 built, 2026-11-07, and fix 5 corrected.** Fix 5 treated two kinds of staging as one. Bytes on their way
to the device take `host_staging_memory_resource` whatever it is: the weight streaming at load in the five
transformers now does, and their CUDA imports and `#ifdef`s are gone. Bytes the host reads -- Gemma's layer scalar,
an archive load's buffer -- need `is_host_accessible`, which Metal's shared memory deliberately is not
(`Metal/MetalBackend.md` 5), so the first build failed on the Mac. They now take `HostReadableStagingMemoryResource`,
beside the traits: pinned on CUDA, the host's own memory where the device's staging memory is not host-readable. So
nothing moves to shared memory on Metal, where fix 5 said the layer scalar would. `memory_separate_from_host` is the
one new trait (CUDA true; the CPU and Metal false), asked by Qwen's table residency and `TokenEmbedding`'s check, and
`EmbeddingTableResidency` has a module of its own. Windows: every target, no warnings; 189 load, archive and embedding
tests pass, two fail, the filed 26B pair. The Mac: builds; `MilaTests` 1752 pass, 3 skip, 7 fail, the filed seven of
change 1; the three other binaries pass. The first Mac run reported a passing build and tested the previous binary: its
script read the exit status of a filter, and now reads the build log.

**Deferred (Todd, 2026-11-07): the Python binding's CUDA-only build** (`Lib/Bindings/Python/CMakeLists.txt:8`).
Recommended when it is taken up: build with whatever backends the build has, `mila.Agent` on the handle as the Mac
path, the per-family sessions left CUDA-only as they retire. It holds up `ROADMAP.md` *Mila on Apple Silicon*'s
criterion that `pip install mila-llm` on the Mac opens a session and generates, and whether that session means
`mila.Agent`'s is to be settled then. Chat and Mila Desktop on the Mac do not wait on it.
