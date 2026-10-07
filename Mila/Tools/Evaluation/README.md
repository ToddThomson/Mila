# Mila/Tools/Evaluation

Paired task evaluations: the same benchmark, the same prompts and the same settings, run on two
engines and compared document by document. Built on EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (pinned, 0.4.13),
since that harness is what makes a score comparable to anyone else's.

The first question it answers: **does Mila's BF16 forward pass score the same as HuggingFace
transformers on the same weights?** That needs no quantization, and it is asked first so that every
later number can be read as the quantization's cost alone. A BF16 gap is an implementation finding,
not a format one.

The design of record is `Mila/Specifications/ModelEval.md`: the phases, the tasks each needs, and
the open decisions. This page is how to run it.

| File | Does |
|---|---|
| `run_arm.py` | Runs one arm, `hf` or `mila`, with every setting fixed and shared |
| `compare_arms.py` | Pairs two arms' samples and writes the comparison |

## What makes the arms comparable

- **The same token ids.** Both arms render each prompt with the reference model's own HuggingFace
  chat template and tokenize it with its tokenizer. The `mila` arm sends those ids to MIS's
  `/v1/completions`, which passes them to the model untouched. Neither MIS's prompt template nor
  Mila's tokenizer is in the comparison. Both are worth measuring, but as separate questions.
- **Greedy decoding, one prompt at a time.** Padding a batch changes a BF16 forward pass enough to
  change a greedy reply, so both arms run at batch size 1.
- **The same stop sequences.** HuggingFace stops at them, and MIS cuts its reply at the first one.
  lm-eval leaves this to the server, so a server that ignored `stop` would score differently from
  the same replies cut correctly.
- **The same day.** Llama 3's template writes today's date into the system header. `compare_arms.py`
  checks that every prompt matches and refuses a pair that does not.

`run_arm.py` writes `environment.json` beside each arm's results: versions, device, seed and what
MIS served.

## Setup

One virtual environment for both arms, on the machine with the GPU:

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # the build for your CUDA
pip install -r requirements.txt
hf auth login                                                          # Llama is gated
```

Accept Meta's licence for `meta-llama/Llama-3.2-3B-Instruct` on HuggingFace first. The `hf` arm
loads the reference weights from it, and both arms use its tokenizer.

The `mila` arm needs MIS (`Mila/Adaptors/Inference/Server`) serving a BF16 build of the same model.
Convert it with `Tools/Converters/Llama`, then package and install it unquantized:

```
ExportArtifact --transcode <converted>.bin <weights>.safetensors
ExportArtifact --package Data/Models/Packages/Llama-3.2-3B-Instruct-bf16 ^
  --weights <weights>.safetensors --tokenizer <tokenizer>.bin ^
  --license Licenses/llama3.2/LICENSE --notice Licenses/llama3.2/NOTICE ^
  --base-model meta-llama/Llama-3.2-3B-Instruct --license-id llama3.2 ^
  --as Llama-3.2-3B-Instruct-bf16
ExportArtifact --install Data/Models/Packages/Llama-3.2-3B-Instruct-bf16
```

## Running

Start MIS on the OpenAI protocol, with room for the longest prompt and reply:

```
set MILA_PROTOCOL=openai
set MILA_MODEL=Llama-3.2-3B-Instruct-bf16
set MILA_CONTEXT_LENGTH=8192
mila-server
```

Then run the two arms and compare them. Each arm writes to `<output>/<arm>` and refuses to write
over an earlier run:

```
python run_arm.py mila --output Data/Evaluation/llama32-3b-bf16
python run_arm.py hf   --output Data/Evaluation/llama32-3b-bf16
python compare_arms.py Data/Evaluation/llama32-3b-bf16/hf Data/Evaluation/llama32-3b-bf16/mila ^
  --report Data/Evaluation/llama32-3b-bf16/comparison.md
```

Before the `mila` arm starts, it checks that MIS answers, names the model it serves, and sends a
token-id prompt to confirm MIS reads it as ids. Add `--limit 20` to either arm for a smoke run, and
use the same limit on both. A limited run checks the setup, never a result.

The default tasks are `ifeval` (541 prompts, instruction following, scored by rules with no judge
model) and `gsm8k_cot_llama` (1,319 grade-school maths problems, eight-shot chain of thought, in the
form Meta reports Llama on). `--tasks` takes any lm-eval task list, and `--include-path` adds a
directory of your own task definitions.

## Reading the comparison

Each row is one metric of one task. **Lost** and **gained** count the documents only the reference,
or only the candidate, got right. **Flipped** is their sum over n. **McNemar p** tests whether the
lost and gained counts are more lopsided than chance. The 95% interval is the interval of the
paired difference. Its width grows with the flip rate: on GSM8K's 1,319 problems with 5% of them
flipped it is about ±1.2 points, and a smaller difference is not resolved.

**Replies** shows how often the two engines wrote exactly the same text, and where the others part.
Two correct BF16 implementations do not agree everywhere: they accumulate in different orders, and
a near-tie between two tokens can resolve either way, after which the replies differ. So expect
agreement well short of 100%, scores whose interval contains zero, and a flip rate that is the
**noise floor between two correct engines**. That floor is the reference point for a quantized
build: its flips above the floor are what the quantization costs.

A BF16 difference whose interval excludes zero, or replies that part within the first few
characters, is a Mila finding. Read the paired samples (`samples_*.jsonl` in each arm) for the
documents that flipped.
