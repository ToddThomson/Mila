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

Also Gemma 4 12B FP4 (2026-10-24, the desktop window): Todd's `gemma-4-12b-it-fp4.json`, installed 2026-10-03,
has no `capabilities` block at all, so the session reports `thinking_capable: false` for a model that thinks.

## Llama 3.2 3B calls the weather tool for a question that needs no tool

`Mila/Apps/Chat/Src/main.cpp` tools-weather prompt @ c8f71f60

Llama 3.2 3B calls `get_weather` for "What is the capital of France?" under the tools-weather prompt (2026-10-24,
the console and the desktop session process alike). Its quoted arguments, which reached the tool as `'Paris`,
were the parser's: single-quoted values are read whole since the change after e08e7802.

## A tool call ran in the desktop window without the person being asked, once

`Mila/Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `approveToolCall` @ a9087567

2026-10-24, the first scripted window run over the real session (Llama 3.2 3B FP4, tools-weather, scratch
store): the first turn's `get_weather` call ran and its result was stored, while the script clicked nothing; a
later call in a new conversation did ask. Not reproduced in five later runs (two window runs of the same
sequence with every page message recorded, two single-turn window runs, one over stdio): each asked. That run's
files were overwritten by the next, so what the page received is not known.

## A client that sends on `turn.end` is refused as busy

`Mila/Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `respond` @ a9087567

`turn.end` is sent from inside the turn and `busy` clears after it, so a `turn.send` written on reading
`turn.end` gets "A reply is still being written" (seen 2026-10-24 from a stdio script). The page waits for
`session.state`, so it does not hit this; `ChatDesktop.md` 4.1 does not say which message a client waits for.

## The session sends a thought break when the model is not thinking

`Mila/Apps/Desktop/Session/Src/Desktop.SessionProcess.ixx` `thoughtBreak` @ a9087567

Gemma 4 12B Q4_0 with thinking off, 2026-10-24: `turn.thought_break` after the tool result, with no thought
text before or after it. `ChatDesktop.md` 4.1 says it means the reasoning resumed. The page shows nothing for it.

## The desktop Models view stays open after a load, and offers Load while one runs

`Mila/Apps/Desktop/Ui/views/models.js` @ a9087567

2026-10-24, the window over the real session: choosing Load leaves the person on the Models view once the model
is loaded, and the row's Load button stays during the load (progress shows above the list).

Also, installing `gpt2-small` from it (2026-10-24): after the install the published list is gone again ("Show
available models"); the list offers `gpt2-small`, a base model the app refuses to chat with, and labels its
quantization "none"; and the installed base model keeps a Load button that can only be refused.
