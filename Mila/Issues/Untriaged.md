# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined at the production release tag**, and anything user-reported is a
pointer to its GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## The footprint gates fail on Linux by a fixed ~0.28 GiB

`Mila/Tests/Dnn/Models/Gemma/GemmaModel.Footprint.Cuda.cpp:301` and the Llama twin `:301` @ 8b3e34b8

`GetRequiredMemory_BoundsActualConsumption`, both families, in the dev container on the 5060 Ti at
context 8192: Gemma predicted 8.314 GiB against 8.604 consumed, residual **0.290 GiB**; Llama
predicted 9.811 against 10.078, residual **0.267 GiB**. The bound is 64 MiB. Prediction equals
`getMemoryStats` in both, and the residual barely moves between two different models, so it reads as
CUDA context plus library overhead rather than a modelling error.

Against it, on Windows: Chat at auto (121856) on the 4070 peaks at 11.00 GiB dedicated with a
90k-token prompt, flat at 10.94 through prefill and decode, shared memory untouched after load. The
whole footprint is taken at load, so depth adds nothing and the auto verdict holds with about a
gigabyte spare -- which reads as the bound being wrong rather than the model.

First run of these gates on Linux with weights present -- the WSL runs skip the model tests for want
of `/Data`, and the container gets them through the bind mount. Windows is green, where
`cudaMemGetInfo` reports dedicated memory only. Open which side is wrong: the 64 MiB bound, or a
footprint model that does not carry the context.

## A malformed Gemma tool call parses as a call with no arguments instead of failing

`Mila/Src/Dnn/Models/Gemma/Gemma.Protocol.ixx:480` (`parseArguments`) @ 8b3e34b8

Seen driving Codex through MIS: the model emitted `call:exec_command{cmd="cat line_count.txt"}`
(`=` and plain quotes, off the trained grammar). `gemma_parse_tool_call` returned
`{'name': 'exec_command', 'arguments': '{}'}` -- the loop breaks at the first key not followed by
`:` and keeps what it has. Codex rejected the empty call and the model retried correctly, so the
flow recovered; a client that executes `{}` would not. Qwen's bridge treats a malformed call as prose.

## Codex reports no model metadata for the model MIS serves

`Mila/Adaptors/Inference/Server` `/v1/models` @ 8b3e34b8

Codex 0.142.5 warns "Model metadata for `gemma-4-12b-it-fp4` not found. Defaulting to fallback
metadata" even with its model set to exactly the id `/v1/models` returns. July's notes said matching
the id cleared it. All three flows passed regardless; unknown what the fallback costs on longer runs.

## The wheel's CUDA runtime dependencies admit 13.0 for a binary built against 13.3

`Mila/Bindings/Package/pyproject.toml:47` / `:55` (`nvidia-cublas`, `nvidia-cuda-runtime`
`>=13.0,<14.0`) @ b5593e32

Noticed while moving the docs to state one CUDA version, the one CI builds with.
`RELEASING.md` (*What the declared toolkit is not*) treats the minor as invisible downstream on
driver minor-version compatibility; that covers the driver, not whether a library built against
13.3 headers loads against the 13.0 runtime libraries pip may resolve. Unverified either way.
