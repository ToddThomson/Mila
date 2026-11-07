# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## A session that fails as it starts hangs silently instead of saying why

`Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` (`run`, the `reader` jthread) @ 0.21.0-dev+126

Found running the packaged session on the Mac, 2026-11-07: it found its prompts against the working directory (fixed the same day in `main.cpp`'s `executable_directory`), so `useActivity( "ask" )` threw before the request loop. The exception left `run` while the reader thread was blocked reading standard input, and the `jthread`'s destructor joined it, so the process answered nothing and printed nothing until its input closed; `main`'s "Fatal error" is reached only then. A stack sample shows the main thread in `~jthread` -> `join` and the reader in `getline`. Not Mac-specific: any throw before the loop does this.

## An edit to a low-level module rebuilds about 530 test targets, which makes a forced failure there cost minutes

`Lib/Src/Dnn/Compute/Registry/DeviceRegistryHelpers.ixx`, `Devices/Device.ixx`; `Lib/Tests` (`import Mila;`) @ 0.21.0-dev+125

Found running change 1's forced failures (`Specifications/Notebooks/DeviceAssumptions.md`): each mutant of a device module rebuilt about 530 `MilaTests` targets, so eleven mutants were estimated at 40 to 50 minutes, and the run was cut to the two that mattered most. Two causes, not yet split by measurement: a module's bodies live in its interface unit, so a body edit changes what every importer sees; and most test files `import Mila;`, so every test depends on every module. `ninja -t query` on the build would show how much is each. Todd, 2026-11-07: noted, moved on. **Direction agreed the same day (Todd):** a module that is not a template, not on a per-token path, and has real bodies or includes a backend header gets an implementation unit in an `Implementation/` folder beside it; value types, accessor classes and templates stay in one `.ixx`; existing code moves when a change touches it substantially. First case: `Agent` and `ModelHandle` in the device-blind handle change, with one body edit's rebuild timed before and after; the convention goes into CLAUDE.md with that number, or is dropped if the saving is not there. **Measured at +127:** an edit to `Agent::baseModelRefusal` rebuilt 263 steps in 268 s in the interface and 19 in 20 s in `Implementation/Agent.cpp`; the convention is in CLAUDE.md. What remains here is the other cause. Separately, tests importing only what they use: white-box tests yes, contract tests and apps stay on `import Mila;`, to be measured after the file-name pass.

## `TPrecision` is named for no one thing, and documented six ways where it is documented at all

`Lib/Src` (about 109 files declare `TensorDataType TPrecision`, about 38 carry a `@tparam` for it); `Specifications/OperationDispatch.md` line 40 @ 0.21.0-dev+125

The decision is `OperationDispatch.md`'s: the activations' element type and the compute precision, the operation owning anything it widens internally, and through `TWeight`'s default the element type of unquantized weights. The `@tparam` lines say "compute and activation precision", "logits precision", "tensor data precision for this component", "abstract tensor precision", "tensor element type and cuBLASLt compute type" and, in `CudaMoeOp`, "activation and weight precision". `Linear` calls the same parameter `TComputePrecision`. No rule asks for `@tparam` lines; CLAUDE.md's Doxygen rule is the file comment's. Todd, 2026-11-07: document it to that meaning and rename it for its context, after the device decisions.

## The ROCm device folder is a Vulkan placeholder, and a ROCm build cannot compile

`Lib/Src/Dnn/Compute/Devices/Rocm/RocmDevice.ixx`, `Registry/DeviceRegistrar.ixx:70`, `Lib/CMakeLists.txt:1268` @ 0.21.0-dev+125

`RocmDevice.ixx` declares `module Compute.VulkanDevice` and a `VulkanDevice` returning `DeviceType::Vulkan`, which `DeviceType` does not have; it is not in the ROCm file set. `DeviceRegistrar` calls `RocmDeviceRegistrar::registerDevices()`, which no file defines. `DeviceType.ixx` still marks Metal and ROCm `FUTURE`.

## Speculative decoding runs on CUDA only: Metal has no multi-token decode outside the expert bank and the sampler

`Lib/Src/Dnn/Components/Linear/Linear.ixx` (`decode`), `Attention/GQA/GroupedQueryAttention.ixx`, `MixtureOfExperts/Router.ixx` (`decodeTokens`), `Transformers/Gemma/Gemma.ixx` (`kDraftModelRuns`) @ 0.21.0-dev+123

Each is `requires ( TDeviceType == DeviceType::Cuda )` or gated by it, and Metal has no `SharedKvAttentionOp`. Metal's expert bank, router and sampler now take several tokens or rows (`Notebooks/MetalGemma.md` 2.1), so a drafter on Metal needs Linear's and attention's multi-token decode and the shared-cache attention. The Q4_0 Metal bank also has one kernel for prefill and decode, where CUDA has a grouped prefill; its speed is unmeasured.

## Gemma 4 12B on CUDA decodes two positions past the sliding window ten times further from HuggingFace than the rest

`Lib/Tests/Dnn/Models/Gemma/GemmaModel.HuggingFaceReference.Cuda.cpp` (`Q4_0_AcrossTheWindowEdge`); `Specifications/Notebooks/MetalGemma.md` 1.3 @ 0.21.0-dev+123

Q4_0 at BF16 on the 5060 Ti, prefill of 993 tokens of a PG-19 book, then 63 decodes teacher-forced across the local layers' window of 1024. Rows 19 and 57 (positions 1011 and 1049) are 2.9e-1 and 3.3e-1 relative L2 from HuggingFace FP32 on the package's decoded weights, and as far from HuggingFace's BF16 run; every other row is at most 3.4e-2, and HuggingFace BF16's worst is 2.5e-2. No argmax changes. Metal's run of the same check, the same ring design, shows no such rows: its worst is 2.7e-3 in FP16 and 2.0e-2 in BF16 (`MetalGemma.md` 2.2), so it is CUDA's own, the fused decode or its ring indexing. Not yet known which.

## Metal loads Q4_0 and six-bit weights only as stored, where CUDA also quantizes them at load

`Lib/Src/Dnn/Compute/Devices/Metal/Operations/Linear/MetalLinearOp.ixx` (`quantize`), `MetalTokenEmbeddingOp.ixx` (`quantize`) @ 0.21.0-dev+123

CUDA's `CudaLinearOp` and `CudaTokenEmbeddingOp` quantize BF16 weights to Q4_0 and to six bits at load (`Int4Packing.ixx`, `Int6Packing.ixx` are the rule). Metal refuses them with the tensor's name. Gemma's published package is stored Q4_0, so the gate does not need it (`Notebooks/MetalGemma.md` section 4).

## Metal's quantized model gate compares two engines' noise, so FP8 misses CUDA's bar by chance

`Lib/Tests/Dnn/Models/LlaMa/LlamaModel.HuggingFaceReference.Metal.cpp` (`Fp8`, `Fp8QuantizedAtLoad`, `Fp8Cache`, fail); `Specifications/Notebooks/MetalQuantizedWeights.md` 2.3 @ 0.21.0-dev+123

The gate holds Metal to "no further from HuggingFace FP32 than CUDA's run of the same weights". Both carry the format's same rounding, so it compares two draws of each engine's small difference: FP8 and the FP8 cache miss on one number each by 1% to 15%, with every argmax and greedy token HuggingFace's, and FP4 passes. Holding Metal to CUDA's run directly fails on FP4, where CUDA's prefill rounds activations to FP8 (`kUseFp8ActivationPrefill`, inferred cause of Metal scoring the sequence 0.012 better). Decided (Todd, 2026-11-06): HuggingFace FP32 on each format's weights decoded to full precision is the reference, with Phase 4's bars (`Metal/MetalBackend.md` section 10). Gemma 4 12B is gated that way first (`Notebooks/MetalGemma.md` 1.1); Llama's FP8 and FP4 tests are not yet moved to it, so they still fail as above. The 8B's FP32 run needs the layer-streamed driver Gemma's capture uses.

## CUDA's quantize at load writes codes for weights that are not finite, where Metal's refuses the tensor

`Lib/Src/Dnn/Compute/Devices/Cuda/Operations/Linear/Kernels/Quantization/CudaFp8WeightQuantization.cu` (`quantize_fp8_per_channel_kernel`), and the FP4 kernel beside it @ 0.21.0-dev+122

Found writing Metal's quantize to CUDA's bytes, from reading, not run. `fmaxf` passes over a NaN, so a row holding one gets the scale of its other weights and the NaN becomes E4M3's not-a-number code; an infinity makes the scale infinite and every code of its row a NaN. Nothing is refused and nothing is logged, so a damaged checkpoint quantizes into a model that runs and is wrong. Metal counts such weights and refuses the tensor by name (`MetalLinearOp::quantize`, `Notebooks/MetalQuantizedWeights.md` 2.1), as its BF16-to-FP16 load already refuses a value FP16 cannot hold.

## Every model load on Metal warns that decode replay is off, which is how Metal is designed to run

`Lib/Src/Dnn/Models/LlaMa/LlamaModel.ixx:546` (`setDecodeReplay( true )`), the same in `GemmaModel.ixx:679` and `QwenModel.ixx:503`; `Lib/Src/Dnn/Core/Models/LanguageModelNetwork.ixx:522` @ 0.21.0-dev+121

Each family turns replay on whatever the device, and on Metal `createDecodeRecording()` returns null by design (`Metal/MetalBackend.md` 4.4), so the first decode of every load logs `[WARN] ... replay of the decode step is off for this network -- it could not be recorded; every decode step is called`. Seen 2026-11-05 on Llama 3.2 3B through `mila-metal-model-timing`. The same words would mean a real failure on CUDA. Not decided: whether a device says it records, or the network tells a null recording from a failed one.

## Llama 3.2 3B at BF16 on Metal scores a sequence further from HuggingFace than either accepted BF16 run

`Lib/Tests/Dnn/Models/LlaMa/LlamaModel.HuggingFaceReference.Metal.cpp` (`Bf16.SequenceLogLikelihoodAndRanges`, fails); `Specifications/Notebooks/MetalFirstModel.md` 2 @ 0.21.0-dev+121

Mean negative log-likelihood over 1024 wikitext tokens: +1.25e-3 from HuggingFace FP32, against a bar of 8.97e-4, the further of HuggingFace's own BF16 (+9.0e-4) and Mila's CUDA BF16 (-5.5e-4). Its logits and greedy pass, at the same distance as the other BF16 runs, and FP16 on the same Mac is within 6.8e-5. The M1 selects FP16, so no model runs this path today. Not looked for: which operation loses it. A first step would compare per-layer outputs against CUDA BF16 on the same tokens.

## Every network chooses its load staging by `#ifdef MILA_HAS_CUDA` though the device traits name it

`Lib/Src/Dnn/Components/Transformers/LlaMa/Llama.ixx` (`loadParameters`); the same in `Gemma.ixx`, `Gemma.Drafter.ixx`, `Qwen.ixx`, `GptTransformer.ixx` @ 0.21.0-dev+120

Each streams its weights through `CudaPinnedMemoryResource` under `#ifdef MILA_HAS_CUDA` and `if constexpr ( Cuda )`, and through `CpuMemoryResource` otherwise. `DeviceTypeTraits<D>::host_staging_memory_resource` already names the staging resource per device (Cuda pinned, Cpu and Metal their own), and `TokenEmbedding` and `Component` use it. `streamTensorBlobs` takes a device index only for pinned memory, which is why the branch exists. Found running Llama on Metal, where the CPU branch serves and Metal's own staging resource goes unused.

## An agent cannot start a process and keep working while it runs

`Lib/Src/Agents/Programs/ProgramContainer.ixx`; `Lib/Src/Agents/Grants/ToolGrant.ixx` (`runProgram`); `Specifications/ContainedPrograms.md` 4 @ 0.21.0-dev+119

Todd, 2026-11-05: the case that matters is long-lived, an agent running a process across its tool calls. `ProgramContainer::run` starts a program and waits for it, and nothing it starts outlives the run, so a dev server cannot stay up while later tool calls test against it, except inside one command. The model sees output only when the run ends, and nothing can be typed into a running program. `run_command` caps a run at 600 seconds. Under the wait, `MxcProgramContainer` already uses `mxc_spawn_json`, reads each output stream on its own thread and polls `mxc_sandbox_try_wait` (`ContainedPrograms.md` 4.8), so MXC gives what a handle would stand on. Not designed: a handle that lasts across calls, how long it may live, when the person is asked, and how a model reads its output and stops it. Windows only, as all of the container is.

## Rope's angle loses accuracy far into a context, on both backends

`Lib/Src/Dnn/Compute/Devices/Cuda/Operations/Encodings/Rope/Kernels/Rope.Angle.cuh`; `Lib/Src/Dnn/Compute/Devices/Metal/Operations/Encodings/Rope/Kernels/Rope.metal` @ 0.21.0-dev+120

Both compute position times frequency in single precision before taking the cosine and sine. Arithmetic, not measured: at position 8192 and a frequency of 1, the angle's own rounding is about 5e-4 radians, and at 128000 about 8e-3. Found writing Metal's Rope test against a double-precision reference, which holds the bar only to position 2048. Reducing the angle before the multiplication would remove it.

## What MLX does on Apple silicon has not been studied for Metal's speed or for work on the CPU

`Specifications/Metal/MetalBackend.md` decisions (GPU only, no CPU compute); an absence -- the spec does not mention MLX @ 0.21.0-dev+117

Todd, 2026-11-04: Metal stays GPU only for now, and MLX may lead to optimizations, or to CPU use that makes sense. Read so far from its docs and Apple's posts only, nothing run: MLX chooses CPU or GPU per operation over unified memory, with its scheduler ordering work across the two; its docs halve a long chain of small operations on an M1 Max by moving them to the CPU. On the M1 Pro in Phase 1, a commit and wait costs about 220 us and a further dispatch about 3 us, on kernels doing almost nothing. Sampling is the first candidate, since it already sits outside the decode step (`DecodeGraph.md` 4.5). Timed on a real decode step, once Phase 4 has one.

## The token loader's test hung the full suite for two and a half hours

`Lib/Tests/Data/Loaders/TokenSequenceLoader.cpp` (`Reset_Functionality`); `Lib/Src/Data/Loaders/TokenSequenceLoader.ixx` (`~TokenSequenceLoader`, `producerThreadFunc`) @ 73e4c0ad

Seen 2026-11-04 in a full suite on the compaction build: the log stopped at this test at 15:56 and the process used no CPU when looked at, 2.5 hours later. It passed in 190 ms in the run before, and 123 repeats alone passed, cut off by the machine's reboot. Read, not proven: the destructor sets `stop_` without holding `mutex_`, so it can fall between the producer's check of `stop_` and its untimed `cv_producer_.wait`; the notify is lost, and `join` never returns. The test destroys the loader just as the producer reaches that wait. The likely fix is to set `stop_` under the lock.

## Compaction summarizes old tool results that could be cut without asking the model

`Lib/Src/Agents/Agent.ixx` `compactConversation` · `Specifications/Agents/Compaction.md` section 4 @ 0.21.0-dev+116

Tool results are most of the tokens in the sessions measured (`Notebooks/CompactionSummaries.md`), and every one but the latest
goes to the model to summarize. Hermes first replaces long old outputs with a stub, with no model call
(`agent/context_compressor.py`, read in part). A step here would sit after clearing reasoning and before the summary: if it
brings the prompt below the trigger, no summary is written, which on Qwen 3.8 could replace minutes of generation. Against it:
it changes the prompt from the first result cut, so what follows is read again, from zero on Qwen; and whether a model reads a
file again after losing it is unmeasured. Measured on the captured sessions first, as its own change (Todd, 2026-11-03: file it).
Beside it: Codex keeps recent user messages word for word along with its summary (`codex-rs/core/src/compact.rs`), where here
only the message being answered and the policy's `kept` list are sure to survive.

## An agent does not know how much of its context it has left

`Lib/Src/Agents/Agent.ixx` `addResult` · `Apps/Server/src/mila_llm_server/conversation.py`, `InferenceServerWire.md` 3.6 @ 0.21.0-dev+116

The model is told nothing of its budget, so it cannot plan around it: whether to read a large file, or finish now. Claude Code 2.1.300
sends `<total_tokens>N tokens left</total_tokens>` after every tool result. The agent could add one line to a result as it renders:
no tool, which would cost a round and be asked only when the model already suspects a problem; no new turn; and only the end of the
prompt changes, so the cached prefix holds. The number should be the reliable depth, not the context limit. It informs and never
enforces: the runtime keeps every limit. Unmeasured whether a model uses it or rushes because of it, every round or only at
thresholds; no evidence found beyond blogs. Measured on the captured sessions, with the pruning entry above (Todd, 2026-11-03: file it).
Master and worker streams would need this and more (`.internal/Ideas/AgentStreams.md`).

## The inference server's streamed chat reply arrives whole, after the model has finished

`Apps/Server/src/mila_llm_server/routes/factory.py` `_stream_chat` · `Specifications/InferenceServerWire.md` 3.5, Phase 4 @ 0.21.0-dev+116

Every chat API, asked to stream, sends an opening chunk and keepalives while the model works, then the reply in one piece
once the turn ends (found by reading it; Todd, 2026-11-03: "It should" stream). So a client shows nothing until the end,
however long the reply, and cannot time a prefill from its first token. Dated `+68`. The spec plans the fix as Phase 4 and
recommends it for v0.22.0, and nothing in `BACKLOG.md` or `Issues/` carried it. The reason it gives, that the parse runs
after the model finishes, no longer holds in the library: `Conversation::ReplyReader` has routed reasoning, answer and calls
as they arrive since Chat moved onto it. The binding does not expose the reader, and it knows no consumer, so that is
where the fix starts.

## Zeroing a view on CUDA writes past the view, and can run past its buffer

`Lib/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Zero.ixx:69` @ 0.21.0-dev+117

`bytes` is `getStorageSize()`, the whole buffer's, and `dst` is `rawData()`, which starts at the view's offset. So a view at offset k zeroes k elements past the buffer's end. Found by reading, while the same line failed a test on Metal. Not run on CUDA, and whether any caller zeroes a view has not been looked at. `Tensor::getStorageSize` has other callers, none checked for views.

## Host accessors on a pinned or managed FP16 or BF16 tensor read elements as `float`

`Lib/Src/Dnn/Tensors/Tensor.ixx:351` `item`, `:386` `operator[]`, `:509` `data`; `TensorHostTypeMap.ixx:80` @ 0.21.0-dev+117

`TensorHostTypeMap` maps FP16 and BF16 to `float`, and the accessors cast the buffer to it, so element i reads 4 bytes at 4i of a 2-byte array. Found by reading while deciding that a Metal tensor has no host accessors (`MetalBackend.md` 5). Not run on CUDA, and which callers use these on a reduced-precision tensor has not been looked at. `Tensor::outputBuffer` uses the same map.

## Clang warns on two operations' `setParameters` and on `TensorOps.Random` reaching a TU-local entity

`Lib/Src/Dnn/Compute/Devices/Cuda/Operations/Convolutions/CudaCausalConv1dOp.ixx:58`, `.../DeltaNet/CudaGatedDeltaRuleOp.ixx:64`, `Lib/Src/Dnn/Compute/Devices/Cuda/Tensors/Operations/CudaTensorOps.Random.ixx:266` @ 0.21.0-dev+114

Clang 22, a library build: `setParameters` overrides but is not marked `override` in both operations
(`-Winconsistent-missing-override`), and instantiating `fill_uniform` and `fill_normal` references the TU-local `cudaMalloc<float>`
and its `devPtr` and `size` (`-Wreference-tu-local-entity-in-other-tu`). 11 warnings, none from Agents. MSVC gives none of them.

## A pnpm project may not build under Mila Desktop's commands, which tell Node to keep links as named

`Apps/Desktop/Session/Src/Tools/Desktop.ProgramSetup.ixx` (`NODE_OPTIONS`) @ 7cf45bdd

Measured 2026-10-31 (`Notebooks/MxcConformance.md` 7.1): in MXC's container Node's resolving of a script's real
path stats the drive's root, which is refused, so every `node` and `npm` run failed; `--preserve-symlinks
--preserve-symlinks-main` lets them run. pnpm lays `node_modules` out as links into its store and relies on real
paths; a pnpm project under those options is not measured. `dotnet build`'s first run in the container also reports
installing a development certificate; where it lands is not measured.

## A command run through the grant reaches every public host or none, never only the hosts a project needs

`Specifications/ContainedPrograms.md` decision 11; an absence -- no allow list by host name @ 7cf45bdd

Todd, 2026-10-31: after phase 3 (network per call), a list of hosts allowed per project in Mila Desktop, and a
question for each new host. MXC's egress rules match address, protocol and port only; names are out of its scope
(`docs/backends/process-container/networking.md`). Its proxy mode (`runtimeConfig.networkProxy`, "Model 2") sends
WinHTTP traffic and `HTTP_PROXY`/`HTTPS_PROXY` to a loopback proxy the host runs, and asks for `egress.default: deny`
with `ingress.default: allow`. Unmeasured: what that ingress opens, and which tools (npm, git, pip, curl, dotnet)
go through the proxy inside the container. The proxy would decide the grant's network, so it is `Lib/Src`.

## Microsoft has not been told of three accesses MXC's policy accepts that no grant gives

`D:\Claude\mxc-cpp` (`policy-conformance-2.ps1`, `-3.ps1`); an absence -- no report made to Microsoft @ 7b5248a2

Measured 2026-10-27 on MXC v1.0.1, Windows 11 build 26300.9711: with the UI on, a contained process reads the
person's screen in every `processContainer.ui.isolation` mode, and no policy field refuses it; and a file with one
name inside a read-write folder and another outside it, or in a denied folder, is accepted for read and write through
its inside name (bubblewrap on Linux the same). MXC's `SECURITY.md` sends security reports to MSRC, never to public
GitHub issues, and Codex ships with the UI on (`codex-rs/mxc-sandbox/src/policy.rs`), so a public issue would tell
everyone first. Todd, 2026-10-27: report after the design spec; Todd files, under his account.

Also measured 2026-10-27 (`ContainedPrograms.md` phase 1): with no folder granted, a contained process reads much of
`HKCU` -- `Environment`, `Software`, `Software\Classes` -- and the policy has no registry field. Todd, 2026-10-31
(`ContainedPrograms.md` decision 8): a known gap for Mila Desktop, and told to Microsoft with the other two.

## On NTFS a contained run walks its folders whole before every run

`Lib/Src/Agents/Programs/SecondNameWatch.ixx` (`watcher_reports_every_name_`) @ 7b5248a2

Todd, 2026-10-31 (`ContainedPrograms.md` decision 7, narrowed): `ReadDirectoryChangesW` on NTFS reports no name made
outside for a file inside, so off ReFS every run walks the granted folders whole -- about 115 ms for 7,500 files,
5.5 s for 358,000 (`Notebooks/MxcConformance.md` section 7). NTFS's change journal records a link added
(`USN_REASON_HARD_LINK_CHANGE`); whether it can be read without an administrator is unmeasured.

## The grant refuses a file with a second name even when every name lies inside the folder

`Lib/Src/Agents/Grants/ToolGrant.ixx` (`requireOneName`) @ 7b5248a2

Todd, 2026-10-27 (`ContainedPrograms.md` decision 5): kept for v0.21.0. `FindFirstFileNameW` and
`FindNextFileNameW` list a file's names, so the grant could accept a file whose names all lie within granted
folders giving at least the access asked, as the container's walk does (`ContainedPrograms.md` 4.3). A project
whose files are linked between two of its own folders is refused today.

## Two installed models that think are recorded as not thinking, so neither app offers their switch

`mila-store show gemma-4-12b-it-fp4` and `mila-store show Qwen3.8-27B-fp4` in Todd's store @ e9c9430f

Seen 2026-10-27: both list their capabilities as `instruct` alone, while the store's `gemma-4-26B-A4B-it-q4_0` and
the scratch store's `gemma-4-12b-it-q4_0` list `thinking` too. `ModelCapabilities::thinking` is read from the
record, so `Agent::thinks()` is false for both whatever the template's switch says, and Chat and Mila Desktop show
no thinking switch for them. Likely packages installed before the manifest carried thinking; not checked against
what the hub publishes now.

## In Get Things Done, Gemma 4 12B looks at the folder before it proposes a plan about half the time

`D:\Claude\filetools\drive_tasks.py` (the session drive's "the first call proposes a plan" check); `Apps/Desktop/Prompts/tasks.json` (`Apps/Chat/Prompts/tasks.json` until +105) @ cc07f776

Measured 2026-10-27 on Gemma 4 12B Q4_0 at the publisher's sampling: the first call was `list_folder` in 2 of 3 runs
on `+103`'s session and 3 of 3 on the session change after it, `propose_plan` otherwise; the plan, its steps and the
edit ran correctly every time. The same model chose both ways on one build, so the check measures the model's choice,
not the session. On `+105`'s session (Mila Desktop on `Mila::Agents` alone, same prompt and tool schemas byte for byte):
`list_folder` first in 4 of 6 runs, then a plan carried out correctly; `propose_plan` first in 1; and once no plan
at all -- three edits, then an answer claiming a README change it never made. Whether the prompt should allow a look first, or the check should, is not decided.

## Off Windows a tool's grant holds no folder, so no file tool can run there

`Lib/Src/Agents/Grants/GrantFileSystem.Posix.ixx`; an absence -- no POSIX file checks behind the grant @ f42c0d95

Todd, 2026-10-27: the grant is Windows-only for now (`ToolCalling.md` 3.4 *Where*). Every POSIX call refuses, so a
grant of folders throws and a tool reaches none. The likely route is Linux's `openat2` with `RESOLVE_BENEATH` from
a held folder descriptor (Linux 5.6, man7.org openat2(2)), which decides in the kernel rather than by a final path,
plus a link-count check for hard links, which the man page does not mention. Unmeasured; macOS not looked at.

## Nobody has read whether Tavily's terms let Mila Desktop keep search results in a saved conversation

`Apps/Desktop/Session/Src/Tools/Desktop.WebSearch.ixx`; an absence -- Tavily's terms of service not read @ da647c29

Found 2026-10-26 adding Tavily web search: a saved conversation keeps every tool result, search results included, and
Tavily's credits page says nothing on keeping or showing results. Brave grants that only on some plans.

## Mila's HTTP transport names itself Mila 0.20 to every server

`Lib/Src/Distribution/Http/CurlHttpTransport.ixx` (`CURLOPT_USERAGENT`, "Mila/0.20") @ da647c29

Found 2026-10-26 writing Mila Desktop's page reader, which fetches through it: the user agent is a written-in
string from the 0.20 cycle, not `Version.txt`'s, so model installs and web pages alike are requested as Mila 0.20.

## Two of the agent's answers to a stop or a cut have no test that fails without them

`Lib/Src/Agents/Agent.ixx` (`respond`, `dropIncompleteLastCharacter`) @ 6f077474

Found 2026-10-25 writing the limit tests: a round with several calls, stopped at its first, gives every call after
it a stopped result -- the tests' model makes one call per round, so nothing reaches them. And a result cut between
two bytes of one character drops the character's leading bytes; the test's result is ASCII, so no cut lands inside
a character.

## GPT-2 cannot be opened by name, because it has no deployment planner

`Lib/Src/Dnn/Models/Gpt/GptModel.ixx` (`load`); an absence -- no `planDeployment`, no `load( path, request )` @ 1f310d50

Found 2026-10-25 building the architecture list (`ModelHandle.md` 3.3): GPT-2 has a model and a tokenizer, but
its one `load` takes a config and no deployment request, so it cannot be an entry and `gpt2-small` cannot be
opened through the model handle. The spec's proof that adding an architecture is one list entry was GPT-2 (H5).

## Llama 3.1 8B FP4 answers as though a tool's result were made up

`Lib/Tests/Agents/Agent.Cuda.cpp` (`askForTheWeather`); scratch-store package `Llama-3.1-8B-Instruct-fp4` @ efa24591

Measured 2026-10-25 through `Mila::Agents::Agent`, the publisher's sampling, seeds 1 to 6, one question about the weather
in Paris and a tool returning "Sunny, 17 degrees Celsius." The rendered prompt is Meta's template: the result is an
`ipython` turn, quoted as the template quotes a string. With the tools in the first user message, the template's
default, it called once every time and then used the result once in six: "This is just an example response and may
not reflect the actual weather", "The function `get_weather` is not defined in the given code snippet". With them in
the system message, two of six; with that and Chat's when-to-call system prompt, one of six, three runs making no
call. Its manifest says `agentic: false`, so the agent's gate does not ask it to. 3.2 3B in the same runs: zero of
twenty-four. Data `D:\Claude\agent\llama_sweep.txt`, `llama_sweep2.txt`.

## Llama 3.2 3B offered a tool in the system message says it will call it and does not

`Apps/Chat/Prompts/tools-weather.json`; scratch-store package `Llama-3.2-3B-Instruct-fp4` @ e8df2aef

Measured 2026-10-25 at temperature 0, ten questions that need no tool and six about the weather now. With the
tools in the system message and the system prompt saying when to call, as Chat now renders them, 3B FP4 and BF16
alike answered all ten directly and called `get_weather` for none of the six: "I'll check the current weather
for Berlin." and stop, or a sentence and then the call, which the Llama grammar reads as text. With the tools in
the first user message, the template's default, it called for all sixteen. Meta's documented 3.2 prompt, with
pythonic calls (llama-models `text_prompt_format.md`), plus the same sentence: ten of ten, three of six. Data
`D:\Claude\llama-tools\` (`mila-3b-*`, `meta-3b-fp4.jsonl`, `chat-live-Llama-3.2-3B-Instruct-fp4.jsonl`).

## Llama 3.1 8B FP4 writes some tool calls in a shape its template does not define

`Lib/Src/Dnn/Models/LlaMa/Llama.ChatProtocol.ixx` (`templateReplyGrammar`); scratch-store package
`Llama-3.1-8B-Instruct-fp4` @ e8df2aef

Measured 2026-10-25 through the server at temperature 0, tools in the system message
(`chat_template_kwargs.tools_in_user_message: false`) and a system prompt saying when to call: of six weather
questions, three became calls, two were written `{"type": "function", "function": "get_weather", "parameters":
{...}}`, and one was a sentence and then the call. The server answered those three as text, as it does any call
that does not parse. The same questions on Q4_0 weights, Mila's and llama.cpp's alike, gave six well-formed calls,
and on FP8 weights quantized on load six as well. In Chat, which now renders that way, two of six (the answers
used the result) and two answered empty. Data `D:\Claude\llama-tools\mis-8b-fp4-guidance.jsonl`,
`mila-8b-fp8-guidance.jsonl`, `chat-live-Llama-3.1-8B-Instruct-fp4.jsonl`.

## Chat's tool definitions are documented as written into the system prompt by Chat

`Apps/Chat/Src/Chat.SystemPrompt.ixx` (`ToolDefinition`'s comment) @ 5e7de360

Since `+94` every family's template places the tools itself, and Chat writes none into the system turn; the
comment still says Chat serializes them, with Llama's pythonic wording. Fixing it brings the module under the
one-type rule, and it holds five types.

## .NET bindings for Mila

an absence -- no file in the repository or in `.internal/` names them @ a8f5360a

Todd, 2026-10-20, while choosing the toolkit for Chat's desktop window (`Specifications/MilaDesktop.md`
section 9): "We do have a future for .NET bindings." Like the Python binding: a projection of the C++ API
(`Lib/Bindings/`).

Specified 2026-10-21 in `Specifications/DotNetBinding.md`: P/Invoke over a C ABI, experimental until
v0.22.0 or later (Todd).

## Mila Desktop's download is unsigned, has no updates and does not say it needs an NVIDIA GPU

`Specifications/MilaDesktop.md` section 9.4; an absence -- no signing, no update channel, no stated
requirement @ c8f71f60

Todd, 2026-10-24: Mila Desktop is the end-user showcase. Since +101 its installer carries the session, the
CUDA libraries it loads and the Visual C++ runtime, so no toolkit is needed. Still missing: the NVIDIA GPU
requirement stated before the download, code signing (SmartScreen warns on an unsigned one), and updates.

## The installed Qwen 3.8 27B FP4 package does not say it can think or call tools

`%LOCALAPPDATA%\Mila\models\qwen3.8-27b-fp4.json` capabilities; an absence -- no `thinking`, no `tools` @ c8f71f60

Found 2026-10-24 driving Mila Desktop's session process: Todd's installed record declares `instruct` and `agentic`
only, so Chat and Mila Desktop run Qwen 3.8 27B FP4 with no reasoning switch and no effort levels, and the
console's status line has no Thinking clause (the phase 2 baseline shows the same). The fp4-cb3 package built
with `--thinking --tools` declares both. Not checked: whether the published package's `mila.json` or only this
install lacks them.

Also Gemma 4 12B FP4 (2026-10-24, Mila Desktop's window): Todd's `gemma-4-12b-it-fp4.json`, installed 2026-10-03,
has no `capabilities` block at all, so the session reports `thinking_capable: false` for a model that thinks.

## A tool call ran in Mila Desktop's window without the person being asked, once

`Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `approveToolCall` @ a9087567

2026-10-24, the first scripted window run over the real session (Llama 3.2 3B FP4, tools-weather, scratch
store): the first turn's `get_weather` call ran and its result was stored, while the script clicked nothing; a
later call in a new conversation did ask. Not reproduced in five later runs (two window runs of the same
sequence with every page message recorded, two single-turn window runs, one over stdio): each asked. That run's
files were overwritten by the next, so what the page received is not known.

## A client that sends on `turn.end` is refused as busy

`Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `respond` @ a9087567

`turn.end` is sent from inside the turn and `busy` clears after it, so a `turn.send` written on reading
`turn.end` gets "A reply is still being written" (seen 2026-10-24 from a stdio script). The page waits for
`session.state`, so it does not hit this; `MilaDesktop.md` 4.1 does not say which message a client waits for.

## The session sends a thought break when the model is not thinking

`Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `thoughtBreak` @ a9087567

Gemma 4 12B Q4_0 with thinking off, 2026-10-24: `turn.thought_break` after the tool result, with no thought
text before or after it. `MilaDesktop.md` 4.1 says it means the reasoning resumed. The page shows nothing for it.

## Mila Desktop's Models view stays open after a load, and offers Load while one runs

`Apps/Desktop/Ui/views/models.js` @ a9087567

2026-10-24, the window over the real session: choosing Load leaves the person on the Models view once the model
is loaded, and the row's Load button stays during the load (progress shows above the list).

Also, installing `gpt2-small` from it (2026-10-24): after the install the published list is gone again ("Show
available models"); the list offers `gpt2-small`, a base model the app refuses to chat with, and labels its
quantization "none"; and the installed base model keeps a Load button that can only be refused.

## A tool that needs a shell has nothing on Windows to hold it to its grant

`Specifications/ToolCalling.md` sections 3.4 and 6; an absence -- no sandbox under the containment interface @ 571b5063

Todd, 2026-10-26: interested in Microsoft's execution containers (MXC) and NVIDIA OpenShell for the agent. OpenShell
sandboxes a whole agent; its Windows design (NVIDIA/OpenShell#2050, RFC 0013, tracked by #4450) runs MXC through
`wxc-exec.exe` -- AppContainer and a low-integrity token, unelevated, Windows 11 build 26100 or later. Per that design:
read-write and read-only paths are set at creation, denied paths "limited / unsupported"; network filtering by host, IP
or CIDR only, `allowedHosts`/`blockedHosts` "not yet enforced on Windows", so OpenShell's host proxy is the real network
enforcer; mapping five policies onto MXC gave 64 access-broadening errors. #2050 closed stale, "not planned",
2026-10-13; #4450 showed 6 of 9 sub-issues done. Fits Mila per tool, not per agent: the loop runs in the model's
process (section 1). Section 6 records MXC kept out for now (Todd, 2026-10-25). Measured 2026-10-26, a standalone
C++23 program calling `mxc_ffi.dll` v1.0.1 (outside the repository): a process container launches in 40-70 ms; reads
and writes outside the one read-write folder and all network were refused, beside the same accesses inside it
allowed; the process runs as the person's own account; with the UI disabled `whoami.exe` fails to start
(0xC0000142), allowed it runs. Unmeasured: whether it holds a grant as completely as section 3.4's file and network
checks describe.

Measured 2026-10-27 (`Specifications/Notebooks/MxcConformance.md`) and designed (`Specifications/ContainedPrograms.md`,
draft): Todd decided the same day that MXC goes into v0.21.0 as an experiment, off by default.

## No CI job builds Mila Desktop's installer

`.github/workflows/desktop.yml`; an absence -- the `windows-installer` job removed @ 793bddb5

Removed 2026-10-26 when the installer began carrying the session (`MilaDesktop.md` 9.4): `npm run build:windows`
refuses to run without `Window/session/` staged from a C++ build, and the app's workflow (`desktop.yml`) has no C++ build. Before
that the job's upload was an installer that ran only the mock. Nothing builds the installer now but a person
with a Release build directory (`Apps/Desktop/README.md`, *Staging it for the window*).

## The contained-run tests fail, not skip, in an MXC build with no `MILA_MXC_LIBRARY`

`Lib/Tests/Agents/ToolGrant.Programs.cpp:282` (`openMxc`, `MxcProgramContainer::open( {} )`) @ 0.21.0-dev+114

With `MILA_ENABLE_MXC=ON` and the variable unset, every `ContainedRunTest` throws in `SetUp()`: "absolute: The filename,
directory name, or volume label syntax is incorrect". The comment there expects the absent container, and the `GTEST_SKIP`
below it never runs. 14 tests in `x64-claude-verify`, the same before and after the 0.21.0-dev+114 file moves.
