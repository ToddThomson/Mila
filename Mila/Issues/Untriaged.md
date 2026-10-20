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

## Gemma 4 writes malformed tool calls and broken code through Mila's sampler that llama.cpp does not on the same weights

`Mila/Src/Dnn/Samplers/` and the Gemma decode path, through the inference server @ d3c1debf

Found 2026-10-19 running the harnesses on Gemma 4 26B-A4B Q4_0 (5060 Ti, files `D:\Claude\gemma26b-gates`). Hermes short
set: Mila 2/8 (FP8 cache, 75,776 auto; six calls that do not parse -- a closing `<|"|>` dropped, `<|"|]`, a plain
quote), llama.cpp b11216 on Google's Q4_0 GGUF with the 8-bit cache 8/8 in 7.9 min, 0 of 13 edits broken. Codex five
turns: 1 of 7 checks with the FP8 cache (patches degenerate into repeated `- ` lines, the summary claims changes never
made), 1 of 7 with BF16 (invalid Python, a 238-command loop, then over the 37,888 window). Gemma 4 12B FP4: 1/8 at
+77 and 3/8 at +76, so not the weight arena; 2/8 on the 26B with every request prefilled whole, so not prefix reuse.
Not the prompt: Mila's render equals llama.cpp's `/apply-template` byte for byte but `<bos>` and thinking. On one
1,415-token t01 prompt greedy decoding is well-formed on both engines, but sampled 20 times at the model's settings
(1.0, top-p 0.95, top-k 64) Mila writes a well-formed patch 11 times and llama.cpp 18. Every Gemma model gate against
HuggingFace skips: `Data/Models/Gemma/gemma4_12b_it_bf16.bin` and `gemma4_26b_a4b_it_qat_bf16.bin` are not on disk.
Not checked: per-token probabilities against llama.cpp, which separates the sampler from numerics greedy hides.
