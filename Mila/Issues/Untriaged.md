# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## .NET bindings for Mila

an absence -- no file in the repository or in `.internal/` names them @ a8f5360a

Todd, 2026-10-20, while choosing the toolkit for Chat's desktop window (`Mila/Specifications/ChatDesktop.md`
section 9): "We do have a future for .NET bindings." Like the Python binding: a projection of the C++ API
(`Mila/Bindings/`).

Specified 2026-10-21 in `Mila/Specifications/DotNetBinding.md`: P/Invoke over a C ABI, experimental until
v0.22.0 or later (Todd).

## An end user cannot install the Mila desktop app without a developer's toolkit

`Mila/Specifications/ChatDesktop.md` section 9.4; an absence -- the installer carries no CUDA runtime @ c8f71f60

Todd, 2026-10-24: the desktop app is the end-user showcase. A person who installs it has no CUDA Toolkit,
so the app ships the CUDA libraries its session loads (cuBLAS, cuBLASLt and the rest), which needs
NVIDIA's CUDA EULA redistribution terms read first. The .NET binding's option A ("the user installs CUDA
13") does not fit an end user. Also needed: the NVIDIA GPU requirement stated before the download, code
signing, an installer and updates.

## The installed Qwen 3.8 27B FP4 package does not say it can think or call tools

`%LOCALAPPDATA%\Mila\models\qwen3.8-27b-fp4.json` capabilities; an absence -- no `thinking`, no `tools` @ c8f71f60

Found 2026-10-24 driving the desktop session process: Todd's installed record declares `instruct` and `agentic`
only, so Chat and the desktop app run Qwen 3.8 27B FP4 with no reasoning switch and no effort levels, and the
console's status line has no Thinking clause (the phase 2 baseline shows the same). The fp4-cb3 package built
with `--thinking --tools` declares both. Not checked: whether the published package's `mila.json` or only this
install lacks them.

## Llama's tool-call arguments keep the quote the model wrote

`Mila/Apps/Chat/Src/Chat.ToolCallParser.ixx` @ c8f71f60

Llama 3.2 3B FP4's call reaches the tool as `{"location":"'Paris"}` (summary "Getting weather for 'Paris"), in
the console before the split and in the desktop session process alike; Toronto came through clean in the same
run. Llama 3.2 3B also calls `get_weather` for "What is the capital of France?" under the tools-weather prompt.
