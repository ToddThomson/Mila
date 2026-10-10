# Mila/Tools/Evaluation

Paired task evaluations: the same benchmark, the same prompts and the same settings, run on two
engines and compared document by document. Built on EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (pinned, 0.4.13) and
Gorilla's [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
(`bfcl-eval`, pinned, 2026.3.23), since those harnesses are what make a score comparable to anyone
else's.

The design of record is `Mila/Specifications/ModelEval.md`: the phases, the tasks each needs, and
the open decisions. This page is how to run it.

The first question it answers: **does Mila's BF16 forward pass score the same as HuggingFace
transformers on the same weights?** That needs no quantization, and it is asked first so that every
later number can be read as the quantization's cost alone. A BF16 gap is an implementation finding,
not a format one.

| File | Does |
|---|---|
| `run_arm.py` | Runs one arm of an lm-eval evaluation -- `hf`, `mila` or `llamacpp` -- with every setting fixed and shared |
| `run_bfcl.py` | Runs one arm of a BFCL evaluation against a server, sending token ids |
| `reference_server.py` | transformers behind a Completions route with MIS's semantics: BFCL's `hf` arm |
| `servers.py` | The preflight every served arm passes, and the environment record each arm writes |
| `run_inspect.py` | Runs one arm of a Hugging Face Hub benchmark through inspect-ai, to the benchmark's own `eval.yaml` |
| `eval_results.py` | Writes the mila arm's Hub benchmark score as the model repo's `.eval_results/<task>.yaml` |
| `compare_arms.py` | Pairs two arms' results, lm-eval's, BFCL's and inspect-ai's, and writes the comparison |

## Arms

An arm is an engine serving a model. Every arm is given the same token ids for every prompt.

| Arm | lm-eval (`run_arm.py`) | BFCL (`run_bfcl.py`) | Hub benchmarks (`run_inspect.py`) |
|---|---|---|---|
| `hf` | lm-eval loads the model itself | `reference_server.py` | inspect-ai loads the model itself |
| `mila` | MIS | MIS | MIS, Chat Completions |
| `llamacpp` | `llama-server` | `llama-server` | `llama-server`, Chat Completions |

## What makes the arms comparable

- **The same token ids.** Every prompt is rendered with the reference model's own HuggingFace chat
  template (BFCL: with BFCL's prompt format for the model) and tokenized with its tokenizer. Served
  arms receive the ids on `/v1/completions` and pass them to the model untouched. Neither MIS's
  prompt template nor Mila's tokenizer is in the comparison; both are separate questions.
- **Greedy decoding, one prompt at a time.** Padding a batch changes a BF16 forward pass enough to
  change a greedy reply, so every arm runs one request at a time: batch size 1 for lm-eval, one
  thread for BFCL (its default is a hundred), one slot for `llama-server`.
- **The same stop sequences.** HuggingFace stops at them; MIS, `llama-server` and the reference
  server cut the reply at the first one. lm-eval leaves this to the server.
- **The same day.** Llama 3's chat template writes today's date into the system header.
  `compare_arms.py` checks that every prompt matches and refuses a pair that does not.
- **The same card.** Any CUDA card will do, but both arms of a comparison run on the same one: two
  GPU architectures differ in the last digits of the same arithmetic. `compare_arms.py` refuses a
  pair whose arms name different cards.

Before a served arm starts, it checks that the server answers, names the model it serves, and sends
a token-id prompt to confirm the server reads it as sent -- not as text, and without adding a BOS
token of its own. Each arm writes `environment.json` beside its results: versions, the card, seed, the
selection, and what the server reports of itself. The `mila` arm's version is the one MIS reports, the
library its binding was built with.

## Setup

One virtual environment, on the machine with the GPU, on Python 3.12: `bfcl-eval` pins NumPy 1.26,
which does not run on 3.13.

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # the build for your CUDA
pip install -r requirements.txt
pip install -r requirements-bfcl.txt                                   # for run_bfcl.py
pip install -r requirements-inspect.txt                                # for run_inspect.py and eval_results.py
hf auth login                                                          # Llama is gated
```

Accept Meta's licence for `meta-llama/Llama-3.2-3B-Instruct` on HuggingFace first. The `hf` arm
loads the reference weights from it, and every arm uses its tokenizer.

The `mila` arm needs MIS (`Mila/Adaptors/Inference/Server`) serving the same model. For BF16, convert
it with `Tools/Converters/Llama`, then package and install it unquantized:

```
ExportArtifact --transcode <converted>.bin <weights>.safetensors
ExportArtifact --package Data/Models/Packages/Llama-3.2-3B-Instruct-bf16 ^
  --weights <weights>.safetensors --tokenizer <tokenizer>.bin ^
  --license Licenses/llama3.2/LICENSE --notice Licenses/llama3.2/NOTICE ^
  --base-model meta-llama/Llama-3.2-3B-Instruct --license-id llama3.2 ^
  --as Llama-3.2-3B-Instruct-bf16
ExportArtifact --install Data/Models/Packages/Llama-3.2-3B-Instruct-bf16
```

The `llamacpp` arm needs a `llama-server` build and a GGUF of the same model.

## Starting the servers

Each served arm talks to its own server; give them different ports, and run one at a time unless
the card holds both. The context length is the longest prompt and reply together: 8192 for IFEval,
GSM8K and BFCL, and RULER's largest band plus 256 for RULER.

Choose the card once and give it to every arm. MIS takes it as `MILA_DEVICE_INDEX`, the `hf` arm and
the reference server as `--device cuda:<index>`, and both count as CUDA does, which can differ from
`nvidia-smi`'s numbering. List the cards by the index they take, with the PCI address that `nvidia-smi`
shows beside each:

```
python -c "import mila; mila.initialize(); print(*mila.cuda_devices(), sep='
')"
```

The `llamacpp` arm records no card: `llama-server`'s own report is not read for one. On a machine with
several cards, start it on the chosen card alone (`llama-server --list-devices` names them), and check
that both arms used the same one.

```
rem MIS
set MILA_PROTOCOL=openai
set MILA_MODEL=Llama-3.2-3B-Instruct-bf16
set MILA_CONTEXT_LENGTH=8192
set MILA_DEVICE_INDEX=0
mila-server                                                     (port 8000)

rem llama-server: one slot, so no request is batched with another
llama-server -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 8192 -np 1 --port 8002

rem the reference, for BFCL's hf arm
python reference_server.py --model meta-llama/Llama-3.2-3B-Instruct --context-length 8192 --port 8001
```

Every setting a `llama-server` run was started with that moves a score -- the KV-cache type, flash
attention, the context -- is in its `/props`, and so in the arm's `environment.json`.

## IFEval and GSM8K

```
python run_arm.py hf       --output Data/Evaluation/llama32-3b
python run_arm.py mila     --output Data/Evaluation/llama32-3b --url http://localhost:8000
python run_arm.py llamacpp --output Data/Evaluation/llama32-3b --url http://localhost:8002
python compare_arms.py Data/Evaluation/llama32-3b/hf Data/Evaluation/llama32-3b/mila ^
  --report Data/Evaluation/llama32-3b/mila-against-hf.md
```

Each arm writes to `<output>/<arm>` and refuses to write over an earlier run. Add `--limit 20` to
every arm for a smoke run; a limited run checks the setup, never a result.

The default tasks are `ifeval` (541 prompts, instruction following, scored by rules with no judge
model) and `gsm8k_cot_llama` (1,319 grade-school maths problems, eight-shot chain of thought, in the
form Meta reports Llama on). `--tasks` takes any lm-eval task list, and `--include-path` adds a
directory of your own task definitions.

## RULER

lm-eval's RULER generates its documents at run time with the tokenizer it is given, so every arm sees
the same ones. Choose the bands with `--metadata`, raise `--max-length` past the largest, and take
the first N documents of every band with `--per-band` (RULER makes 500 a band; `--limit` would only
reach the shortest):

```
python run_arm.py hf   --output Data/Evaluation/llama32-3b-ruler --tasks ruler ^
  --metadata "{\"max_seq_lengths\": [4096, 8192, 16384, 32768]}" --max-length 33024 --per-band 100
python run_arm.py mila --output Data/Evaluation/llama32-3b-ruler --tasks ruler ^
  --metadata "{\"max_seq_lengths\": [4096, 8192, 16384, 32768]}" --max-length 33024 --per-band 100 ^
  --url http://localhost:8000
```

Start the server with a context of at least the `--max-length`. Each band is reported as its own
row. Four of the 13 tasks fetch text at run time (Paul Graham's essays, SQuAD, HotpotQA).

## BFCL

`run_bfcl.py` drives `bfcl-eval` against a server -- the reference server for `hf`, MIS for `mila`,
`llama-server` for `llamacpp` -- and evaluates the replies. BFCL renders each entry in the model's own
function-calling format; `run_bfcl.py` tokenizes it with BFCL's own HuggingFace tokenizer and sends
the ids, at temperature 0:

```
python run_bfcl.py hf   --output Data/Evaluation/llama32-3b --url http://localhost:8001
python run_bfcl.py mila --output Data/Evaluation/llama32-3b --url http://localhost:8000
python compare_arms.py Data/Evaluation/llama32-3b/hf Data/Evaluation/llama32-3b/mila
```

The default model is BFCL's `meta-llama/Llama-3.2-3B-Instruct-FC` handler and the default selection
its `python` collection: every single-turn Python category, live and not, about 3,500 entries.
`--categories` takes BFCL's category or collection names, and `--limit 5` the first five entries of
each category for a smoke run. Results go to `<output>/<arm>/bfcl`, beside the arm's lm-eval results,
and `compare_arms.py` reads both.

## Hugging Face Hub benchmarks

A benchmark registered on the Hub carries an `eval.yaml` that fixes its prompt, solver and scorer. A model repo
reports a score against it in `.eval_results/<task>.yaml`, and the Hub shows that score on the model page and in the
benchmark's leaderboard. Every registered language benchmark runs on inspect-ai, and lm-eval is not among the
frameworks the Hub accepts, so a score that belongs on such a leaderboard comes from `run_inspect.py`, not
`run_arm.py`:

```
python run_inspect.py hf   --output Data/Evaluation/llama32-3b-hub
python run_inspect.py mila --output Data/Evaluation/llama32-3b-hub --url http://localhost:8000
python compare_arms.py Data/Evaluation/llama32-3b-hub/hf Data/Evaluation/llama32-3b-hub/mila
python eval_results.py Data/Evaluation/llama32-3b-hub/mila --model-repo <clone of the model's Hub repo> ^
  --notes "FP4 weights"
```

The default is MMLU-Pro (`TIGER-Lab/MMLU-Pro`, about 12,000 questions, zero-shot, the model writes its answer
letter); `--benchmark` takes another registered benchmark's dataset id and `--task` one of its tasks. Each arm
pins the revision of the definition it ran, and decodes greedily, one request at a time, with the same
`--max-tokens` on every arm (2,048 by default).

This differs from the other harnesses in one control: inspect-ai sends chat messages, not token ids, so each
engine renders the prompt in its own template. The mila arm therefore measures MIS's prompt as a user meets it,
including the system turn MIS adds when a request has none, and a comparison against `hf` measures the engine and
the template together. `compare_arms.py` still refuses arms whose benchmark inputs differ.

`run_inspect.py` refuses a benchmark that is not registered (IFEval has no `eval.yaml`), one scored by a judge
model (the Hub's GSM8K is: `model_graded_fact`, which with no grader named grades with the model under test), and a
gated one whose terms the logged-in account has not accepted (GPQA and HLE). `eval_results.py` refuses a run that
used `--limit`, an arm other than `mila`, and a file that already exists. Its value is a percentage, as the Hub's
leaderboards are written.

A gated benchmark's terms forbid publishing its examples, and inspect-ai's logs quote every question. For such a
benchmark, `eval_results.py` links a source only with `--source-private`, which states that the source is readable
only by people who accepted the terms.

## Reading the comparison

Each row is one metric of one task (for RULER, one band; for BFCL, one category). **Lost** and
**gained** count the documents only the reference, or only the candidate, got right. **Flipped** is
their sum over n. **McNemar p** tests whether the lost and gained counts are more lopsided than
chance. The 95% interval is the interval of the paired difference. Its width grows with the flip
rate: on GSM8K's 1,319 problems with 5% of them flipped it is about ±1.2 points, and a smaller
difference is not resolved.

**Replies** shows how often the two engines wrote exactly the same text, and where the others part.
Two correct BF16 implementations do not agree everywhere: they accumulate in different orders, and
a near-tie between two tokens can resolve either way, after which the replies differ. So expect
agreement well short of 100%, scores whose interval contains zero, and a flip rate that is the
**noise floor between two correct engines**. That floor is the reference point for a quantized
build: its flips above the floor are what the quantization costs. A BF16 GGUF on the `llamacpp` arm
measures the floor a second time.

A BF16 difference whose interval excludes zero, or replies that part within the first few
characters, is a Mila finding. Read the paired results (`samples_*.jsonl`, and BFCL's
`result/` and `score/` files) for the documents that flipped.
