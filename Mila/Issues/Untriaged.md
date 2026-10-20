# Untriaged

Captured, not yet judged. Writing here needs no judgement, which is the whole point — finding
something mid-task leaves seconds, and a format that asks for more is a format that goes unused.
Facts may grow; judgement may not.

Entries are **deleted unexamined after ninety days**, and anything user-reported is a pointer to its
GitHub issue rather than a copy. Triage flow, categories and the entry format are in
[README.md](README.md); entries here carry an anchor where the judged files carry tags.

---

## The inference server logs a client's dropped connection on Windows as an error with a traceback

`Mila/Apps/Server/src/mila_llm_server/app.py` (event loop) @ d3c1debf

Found 2026-10-19 in the Codex session against Qwen 3.8 fp4-cb3: between two requests asyncio logged `ERROR Exception
in callback _ProactorBasePipeTransport._call_connection_lost()` with a `ConnectionResetError` [WinError 10054]
traceback, and the next request was served normally; Codex's five turns all completed. It is the Windows proactor
loop reporting a peer that reset an idle keep-alive connection. Not checked: whether uvicorn's own loop setting or a
handler for that one exception removes it, and whether it appears on Linux.

## Three Gemma 26B-A4B tests fail on a 32 KB stub of its BF16 weights instead of skipping

`Data/Models/Gemma/gemma4_26b_a4b_it_qat_bf16.bin`, `Mila/Tests/Dnn/Models/Gemma/GemmaModel.MixtureOfExperts.Load.Cuda.cpp:76` @ 2a673ec5

Found 2026-10-20 in the full suite: `Fp4Load_...`, `Q4_0Load_...` and `GemmaMixtureOfExpertsParityCudaTests.
LayerStream_MatchesHuggingFaceReference` fail in milliseconds with "Failed to read rank of 'tf_layer_24...'". The file
is 32,734 bytes, written 2026-10-19 22:44 by a 26B conversion that crashed (0xC0000005 in 5 s, empty log,
`D:\Claude\gemma26b-gates\convert-26b.log`) while restoring the Gemma BF16 checkpoints; the 12B's re-conversion
succeeded. The tests gate on the file existing, so a stub turns a skip into a failure. Not checked: why the
converter crashed (the HF cache holds the 49 GB QAT checkpoint), and whether a converter that dies should leave no
file behind.
