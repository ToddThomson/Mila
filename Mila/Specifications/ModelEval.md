# Model Evaluation

**Status:** Draft, 2026-10-07. Agreed in discussion with Todd the same day: start with BF16 parity on Llama 3.2 3B
over IFEval and GSM8K, then add BFCL, RULER and a llama.cpp arm. The tools for Phases 1 to 4 are built
(`Mila/Tools/Evaluation`, `0.21.0-dev+41` and `+42`) and rehearsed end to end on a stand-in model. 2026-10-10
(`+55`): Phase 1's two arms ran on real weights on CUDA for the first time, a 20-document smoke run per task on one
card, which checks the setup and is not a result. No release admits this work; section 10 holds the decisions that would. Added 2026-10-08: scores
for the Hugging Face Hub's benchmark leaderboards (section 11, Phase 7), built at `0.21.0-dev+43` and rehearsed
against a stand-in server on MMLU-Pro's own definition.

**Area:** what Mila states about a model's quality on standard task benchmarks, and how such a statement is made
comparable to anyone else's. Its uses: the "is it any good" answer on every model card; the measured cost of each
weight format, read against a measured noise floor; and evidence that Mila's forward pass is correct at the scale of
a whole benchmark, not one prompt.

---

## 1. The question

A user deciding on a quantized model asks three things, and each has a different kind of answer:

| Claim | Question | Unit | Owner |
|---|---|---|---|
| **Fidelity** | How close is this build to the original? | KL from BF16, top-1 agreement, perplexity ratio, flip rate | `Quantization.md` (teacher-forced); this spec (flip rate) |
| **Capability** | Is it good at tasks people recognise? | Benchmark score, as a difference from the original under one harness | This spec |
| **Performance** | Is it fast on my card? | Prefill and generation tokens a second, footprint at a context | `benchmark_comparison.py` (`BACKLOG.md`, the llama.cpp comparison) |

Depth is the fourth question and is not this spec's: where in a long conversation a configuration stops finding
facts, obeying instructions and calling tools is `ContextProfile.md`. The two meet in section 5's tool-call and
long-context tasks, which are the standard counterparts of its arms.

What Mila had before this spec is fidelity alone: perplexity ratios against a reference build
(`ModelCards/Qwen3.8-27B-cb2-3`), KL and top-1 tables (`Quantization.md`). They are honest and specific, and a
user cannot act on them: "13.9% higher perplexity" does not say whether the model still does the job. A score on a
benchmark the reader already knows does.

## 2. Principles

1. **Never compare to a vendor's published score.** Meta, Google and Alibaba report under their own prompts, shot
   counts and templates; a Mila build measured any other way differs from their number for reasons unrelated to
   the build. Every claim is a difference between two arms run under the same harness, on the same day.
2. **The reference is an engine the reader trusts, not Mila.** HuggingFace transformers at BF16 is the anchor. Mila
   BF16 against it is the first measurement (section 7, Phase 1), because until that difference is known, a
   quantized build's difference cannot be attributed to the quantization.
3. **Paired, not averaged.** Both arms score the same documents, so a comparison counts the documents that changed,
   in each direction, and tests whether the imbalance is larger than chance (section 6). Two averages and two
   standard errors would hide that half the answers changed.
4. **The noise floor is measured, not assumed.** Two correct BF16 engines disagree on some replies (section 6.3).
   That disagreement is the floor, and a format's cost is what it adds above it.
5. **The harness is the public one.** EleutherAI's lm-evaluation-harness, pinned, is what the field reports
   through. Mila owns the arm definitions and the comparison, never a task's prompt or scoring.

## 3. Arms

An **arm** is an engine serving a model, run under the fixed settings of section 4.

| Arm | Engine | lm-eval tasks (`run_arm.py`) | BFCL (`run_bfcl.py`) | Hub benchmarks (`run_inspect.py`) |
|---|---|---|---|---|
| `hf` | transformers, BF16 | lm-eval's `hf` model | `reference_server.py` | inspect-ai's `hf` model |
| `mila` | MIS on the OpenAI protocol, any store variant | `local-completions` | MIS | MIS, Chat Completions |
| `llamacpp` | `llama-server` with a GGUF | `local-completions` | `llama-server` | `llama-server`, Chat Completions |

lm-eval loads the reference model itself; BFCL only talks to a server, so its `hf` arm is `reference_server.py`:
transformers behind a Completions route with MIS's semantics there -- text or token-id prompts, greedy only, `stop`
honoured, the end-of-turn token dropped and every other control token kept, as Mila's decode keeps them. Run on the
same model behind MIS's real routes, the two wrote byte-identical replies to every BFCL entry in the rehearsal.

Mila reaches the harness through MIS, not the Python binding. MIS is the surface users call, so the measurement
exercises it, and every capability the harness needs that MIS lacks is a defect in the adaptor, not a feature
request -- the library already holds it (`CLAUDE.md`: an adaptor adds no capability). The binding stays
consumer-blind and gains nothing for this spec.

## 4. Controls

Each control removes one variable that is not the engine. All are enforced by `run_arm.py` or checked by
`compare_arms.py`; none is a convention a person must remember.

- **The same token ids.** Every arm renders each prompt with the reference's own HuggingFace chat template -- or,
  for BFCL, with BFCL's own prompt format for the model -- and tokenizes it with the reference's tokenizer. Served
  arms receive the ids on `/v1/completions`, which MIS passes to the model untouched (added at `+41` for this).
  BFCL sends text; `run_bfcl.py` replaces its request with one carrying the ids. MIS's own prompt template and
  Mila's tokenizer are therefore outside the comparison; each is its own question (section 9). Every served arm
  proves the path before a run (`servers.py`): it sends a rendered probe and refuses to start unless the server
  reports exactly the probe's token count, which also catches a server that adds a BOS token of its own.
- **Greedy, one prompt at a time.** Batch size 1 on every arm, and one request in flight: BFCL's default of a
  hundred concurrent requests is set to one, and `llama-server` runs one slot (`-np 1`; the arm warns otherwise). A
  padded or shared batch changes a BF16 forward pass enough to change a greedy reply.
- **Stop sequences honoured by the server.** lm-eval leaves `stop` to an API server and never cuts a reply itself,
  so a server that ignored it scores differently on the same replies. MIS cuts a buffered reply at the first stop
  sequence (`+41`). A streamed reply does not, and no arm streams.
- **The same prompts, checked.** lm-eval logs a hash of each prompt. `compare_arms.py` refuses two arms whose
  documents or prompts differ. Llama 3's template writes today's date into every prompt, so both arms run on the
  same day.
- **The same length budget.** `max_length` 8192 on every arm by default, raised past RULER's largest band for
  RULER; lm-eval left-truncates a context past it identically.
- **The same card.** Any CUDA card, the same one for both arms: Ada and Blackwell differ in the last digits of the
  same arithmetic, so a pair across cards adds the card to the engine difference. Each arm records its card by name
  and PCI address -- an index alone is not one, since CUDA's numbering and `nvidia-smi`'s differ -- and
  `compare_arms.py` refuses arms that name different cards. MIS reports its card and the version of the library its
  binding was built with on `/v1/models` (`0.21.0-dev+55`), from `mila.cuda_devices()` and `mila.version()`.
- **A recorded environment.** Each arm writes `environment.json`: Mila version, harness and transformers versions,
  torch and the card for `hf`, what the server reports of itself for a served arm -- `llama-server`'s `/props` gives
  its build, weights file, context and KV-cache settings -- and the seed, tasks and selection.

One difference is not controlled. MIS's Llama model and `llama-server` each reuse the KV state of a prompt prefix
they have already seen, so a few-shot task's shared prefix is computed once and reused; transformers recomputes it.
Reuse is each engine's own behaviour as a user meets it, and a correct reuse changes nothing. If an arm's replies
part from the reference at the end of a shared prefix, reuse is the first suspect.

## 5. Tasks

A task needs one of two things from an arm: generation (`generate_until`), or the log-likelihood of a given
continuation (`loglikelihood`, which multiple-choice tasks use). MIS serves the first. The second needs
`logprobs` and `echo` on `/v1/completions`, which MIS does not serve and the library already computes
(`sequenceLogLikelihood`, all three families) -- the adaptor gap section 7, Phase 5 closes.

| Task | Measures | Needs | In phase |
|---|---|---|---|
| `ifeval` | Instruction following, 541 prompts, rule-scored with no judge model | generation | 1 |
| `gsm8k_cot_llama` | Grade-school maths, 1,319 problems, eight-shot chain of thought in Meta's form | generation | 1 |
| BFCL, `python` collection | Function calling, single turn: simple, multiple, parallel, irrelevance, live and not; about 3,500 entries | generation | 4 |
| RULER | Retrieval, tracing and aggregation at length, 13 tasks per band | generation | 4 |
| `mmlu_pro` | Knowledge and reasoning, ten-way multiple choice | generation (CoT form) or log-likelihood | 5 |
| `arc_challenge`, `hellaswag`, `winogrande` | The classic quantization-paper set | log-likelihood | 5 |

Phase 1's two were chosen because they are generative (no MIS change beyond section 4), cheap enough to run in hours
on a 3B model, and sensitive where quantization damage shows first: multi-step arithmetic and exact
instruction compliance.

BFCL and RULER are the recognised counterparts of `ContextProfile.md`'s tool-call and recall arms. Reporting both
lets a reader check the in-house measurement against one they already trust.

**BFCL** is Gorilla's `bfcl-eval` (pinned, 2026.3.23), not part of lm-eval. Its handler for a model renders the
function declarations into that model's own prompt format and checks the reply's calls against the answer by
syntax tree; for Llama 3.2 3B that is `meta-llama/Llama-3.2-3B-Instruct-FC`. It measures the model's tool calling
from a prompt BFCL renders, not MIS's tool path: MIS's grammars and its Responses route are not exercised, and are
a separate question (section 9). The default selection is BFCL's `python` collection, every single-turn Python
category; multi-turn, memory and web search are its agentic categories, and web search needs the network during a
run. `compare_arms.py` reads BFCL's per-entry results: an entry passed unless its category's score file lists it.

**RULER** is lm-eval's port (`ruler` group, 13 tasks). It generates its documents at run time with the tokenizer
it is given, so every arm sees the same documents. Its bands are set with lm-eval's metadata
(`max_seq_lengths`), and it generates 500 documents per band, laid end to end; `--limit` would take only the
shortest band, so `run_arm.py --per-band N` selects the first N of every band. Defaults set with the tool: bands
of 4K, 8K, 16K and 32K, and 100 documents a band, revisited after the first run. 64K is the edge of the 16 GB card
for the `hf` arm (Llama 3.2 3B's BF16 cache is about 114 KB a token, 7.3 GB at 64K, beside 6.4 GB of weights), and
128K does not fit it; past what the reference holds, a band has no reference, and Mila against itself across
formats is `ContextProfile.md`'s. Four of the 13 tasks fetch text at run time (essays and two QA sets).

**Not in scope:** benchmarks scored by a judge model (MT-Bench, Arena-Hard), since the judge becomes a third engine
in the comparison; and code benchmarks that execute generated code (HumanEval, MBPP), until there is a sandbox to run
them in.

## 6. Statistics

`compare_arms.py` computes everything here from the paired samples; nothing is read off lm-eval's own summary.

### 6.1 Per metric

For a metric scored 0 or 1 per document (or per instruction, for IFEval's instruction-level accuracy, paired by
position), over n paired outcomes:

- **Lost** (b): documents only the reference got right. **Gained** (c): only the candidate.
- **Difference**: (c - b) / n, candidate minus reference.
- **95% interval**: difference ± 1.96 · sqrt((b + c) - (c - b)² / n) / n, clamped to [-1, 1]. The width grows
  with the flip rate: GSM8K with 5% of its problems flipped resolves about ±1.2 points.
- **Flip rate**: (b + c) / n.
- **McNemar p**: exact two-sided binomial test of b against b + c at one half.

A metric on a scale takes the mean paired difference and its interval from the standard deviation of the
differences.

### 6.2 Replies

Per task, the fraction of documents on which both arms wrote byte-identical text, and for the rest the median number
of characters before they part. Agreement is a diagnostic, not a claim: it locates divergence, and a median near zero
means the arms differ from the first tokens, which a correct engine should not.

### 6.3 The noise floor

Two correct BF16 implementations accumulate in different orders. A near-tie between two tokens can then resolve
either way, and greedy decoding diverges from there. So BF16 against BF16 is expected to show agreement short of
100%, a non-zero flip rate, and a difference whose interval contains zero. **That flip rate is the floor.** A
quantized build is compared against the same reference, and what it adds above the floor -- in flips, and in the
direction of the lost-gained imbalance -- is the format's cost.

This is the reason a quantized number is reported beside its floor and never alone. The direction matters as much
as the count: equal accuracy with a high flip rate is a model that answers differently, which the accuracy hides
(Dutta et al., "Accuracy is Not All You Need", 2024).

## 7. Phases

Each phase states its gate before its run. A result goes to a notebook (`Notebooks/ModelEval.md`, created with the
first run), and any decision it forces comes back here.

### Phase 1 -- BF16 parity, Llama 3.2 3B

Mila BF16 (`Llama-3.2-3B-Instruct-bf16`, installed from a converted package) against transformers BF16 on
`meta-llama/Llama-3.2-3B-Instruct`, IFEval and GSM8K in full.

**Gate:** every metric's 95% interval contains zero, and every task's median divergence is past the first sentence.
A failure is a Mila implementation finding, to be localised from the flipped documents' samples before anything
else in this spec proceeds.

**Built:** the MIS changes (token-id prompts, `stop`), `run_arm.py`, `compare_arms.py`. Rehearsed on CPU with a
random-weight Llama-shaped model behind MIS's real routes: prompts paired, replies identical, a planted difference
counted and tested, a planted prompt mismatch refused.

### Phase 2 -- Quantized builds against the floor

The published FP4 build of the same model, and FP8, each as a `mila` arm against the same `hf` reference, beside
Phase 1's floor. Their difference from the floor is each format's cost on this model.

### Phase 3 -- llama.cpp

A `llamacpp` arm serving the Q4_K_M and Q4_0 GGUFs, through `llama-server`'s OpenAI endpoint with token-id prompts
-- the same comparison the performance page makes (`BACKLOG.md`), so quality and speed come from one reference. A
BF16 GGUF as a third BF16 arm measures the floor a second time: if llama.cpp BF16 and Mila BF16 flip about as many
answers against transformers, the floor is real; if only Mila's is high, that is a Mila finding.

**Built:** the `llamacpp` arm in `run_arm.py` and `run_bfcl.py`, with `llama-server`'s `/props` recorded. Not
rehearsed here: no `llama-server` in this environment. Its preflight is the same probe as MIS's.

### Phase 4 -- BFCL and RULER, Llama 3.2 3B

Every arm of Phases 1 to 3 on BFCL's `python` collection and on RULER at section 5's bands.

**Gate:** as Phase 1's for the BF16 arms, per BFCL category and per RULER band.

**Built:** `run_bfcl.py`, `reference_server.py`, `servers.py`, `run_arm.py --per-band`, and both readers in
`compare_arms.py`. Rehearsed on the stand-in model: BFCL's eleven `python` categories, three entries each, against
`reference_server.py` and against MIS's routes, every prompt paired and every reply identical; RULER's synthetic
tasks at two short bands.

### Phase 5 -- Log-likelihood tasks

`logprobs` and `echo` on MIS's `/v1/completions`, projecting `sequenceLogLikelihood`, then the multiple-choice
tasks of section 5. Also gives every log-likelihood task a token-exact comparison without generation, which is a
sharper parity test than Phase 1's.

### Phase 6 -- Every published model

Llama 3.1 8B, Gemma 4 12B and 26B-A4B, and both Qwen 3.8 builds, on the tasks of Phases 1, 4 and 5. Blocked on
section 10's reference decision for every model whose BF16 weights exceed the card.

### Phase 7 -- Hub benchmarks

MMLU-Pro through `run_inspect.py` on the `hf` and `mila` arms of Phase 1's model, compared by `compare_arms.py`, and
the mila arm's score written to the model repo by `eval_results.py` (section 11). GPQA follows once its definition
has been read (section 11.3).

**Gate:** none for publishing, which is section 10's decision 5. The comparison against `hf` is read as Phase 1's
is, knowing it also measures MIS's prompt template (section 9).

**Built:** `run_inspect.py`, `eval_results.py`, and an inspect-ai reader in `compare_arms.py`. Rehearsed against a
stand-in server on MMLU-Pro's real `eval.yaml`, with stand-in rows in its schema (the container could not fetch
the Hub's data files): two served arms paired and compared, the result file written, and every refusal of section
11.4 exercised.

## 8. On the model card

The card answers "is it any good" with this measurement and nothing else on its first screen (`CLAUDE.md`,
*End-User Prose*):

- One table: rows are tasks, columns are **Original (BF16)**, **This model**, and the llama.cpp build of the same
  size where Phase 3 has run. Scores as percentages from the same harness.
- One sentence naming the largest difference and whether it is larger than the benchmark resolves.
- Below its own heading: the flip rate against the floor, the harness version and settings, and the fidelity
  numbers `Quantization.md` owns.

The reader never sees "arm", "noise floor" or "paired" on the first screen; the table is the claim. The comparison's
JSON is the source for both the card and the website, so a number is written once.

A Hub benchmark score (section 11) also appears on the model page, from the repo's `.eval_results/`, beside the
card rather than in it. It answers a different question -- how this model ranks against others under the
benchmark's own protocol -- and it is not the card's claim.

## 9. Separate questions

Each was excluded from the comparison on purpose and is worth measuring on its own terms:

- **MIS's prompt template.** Through `/v1/chat/completions`, MIS renders Llama's prompt itself, and differently from
  Meta's (`Untriaged.md`). The same task through chat completions against Phase 1's run measures what that costs.
- **Mila's tokenizer.** Mila's BPE against HuggingFace's on every Phase 1 prompt: identical ids, or the first
  difference.
- **MIS's tool path.** BFCL measures tool calling from a prompt it renders itself. A user's agent goes through MIS's
  Responses or Messages route, which renders the declarations and parses the calls in each family's own grammar;
  BFCL's categories driven through that route, against Phase 4's run, measure what the route costs.
- **Sampling.** Every arm here is greedy. A user samples; a sampled evaluation needs repeats and a different
  statistic, and is not planned.

## 10. Open decisions

1. **The reference for models whose BF16 weights exceed the 16 GB card.** Llama 3.1 8B at BF16 is about 16 GB of
   weights, Gemma 4 12B about 24, Qwen 3.8 27B about 54: none runs on the card the project measures on. Options: a
   larger card for the `hf` arm alone (a rented one is enough -- the reference does not need Mila); transformers
   with CPU offload, correct but slow; or Mila FP8 as a proxy reference, which is cheap but makes the anchor a Mila
   build, against principle 2.
2. **The thresholds a published format must meet**, as a difference against the reference beyond the floor. Not
   set until Phases 1 and 2 have produced a floor and a cost to set them against.
3. **Admission.** Whether any phase is v0.21 work. The nearest criterion is `ROADMAP.md`'s "every model the release
   publishes is measured for agentic work", which BFCL and RULER would support; Phases 1 and 2 serve the model cards
   and have no criterion of their own yet.
4. **Where the run data lives.** `Data/Evaluation/` is gitignored. A number on a card cites a run that should be
   retrievable later -- a release asset, or a committed JSON summary beside the card. A gated benchmark's run (GPQA,
   HLE) cannot be public at all: its logs quote the questions its terms forbid publishing (section 11.3).
5. **Whether Mila publishes Hub results, and for which models.** Every result on the leaderboards read so far is
   self-reported under its author's own settings (section 11.2), which is the comparison principle 1 refuses. A
   Mila result is reproducible and states its settings, but it sits in a table that is not. Options: publish for
   every model and let the notes carry the settings; publish only where the card's paired comparison already
   stands beside it; or not at all.
6. **Verification.** A verified result needs the run made in HF Jobs (section 11.1), which would need MIS and its
   weights in a Hugging Face job, and Hugging Face's word that a token is issued for an engine it does not host.
   Not pursued until decision 5 is made.

## 11. The Hugging Face Hub

Read 2026-10-08 from the Hub's documentation, the benchmarks' own definitions and their leaderboards. The feature is
marked a work in progress, so this section records what was true that day.

### 11.1 How a result reaches a leaderboard

The Open LLM Leaderboard is archived. Its successor is decentralised: a dataset repo registered as a **benchmark**
holds an `eval.yaml` defining how it is run, and a model repo reports a score against it in
`.eval_results/<task>.yaml`. The Hub shows that score on the model page and aggregates it into the benchmark's
leaderboard. An entry needs the benchmark's dataset id, a task id from its `eval.yaml`, and a value; it may carry the
benchmark revision, a date, a source link and free-text notes.

A result is **verified** only with a token from a run of inspect-ai in HF Jobs. A result in the model repo is the
author's own; one offered by pull request to someone else's repo shows as community-provided while the request is
open. Registering a new benchmark is by request, onto an allow-list.

### 11.2 What that means for Mila

- **The framework is fixed by the benchmark, and it is not lm-eval.** `eval.yaml` names one framework from a list
  the Hub maintains; lm-evaluation-harness is not on it, and every registered language benchmark read so far uses
  inspect-ai. A `run_arm.py` score filed against a Hub benchmark would be a different measurement from its
  neighbours, so the Hub path is its own runner, `run_inspect.py`.
- **The leaderboards are not controlled comparisons.** MMLU-Pro's held 141 results and GSM8K's 18; none was
  verified, and their notes show settings from greedy to sampled at temperature 1 with long thinking budgets. A
  Mila result there is one reproducible row among rows that are not, which is why section 10 asks whether to
  publish.
- **Values are percentages.** The documentation's examples show both scales; the leaderboards are written in
  percentages (one MMLU-Pro row reports a fraction and ranks near the bottom for it).
- **The prompt is the engine's.** inspect-ai sends chat messages, so each arm renders its own template, and the
  section 4 control of identical token ids does not hold. inspect-ai sends one user message and no system message;
  MIS then adds "You are a helpful assistant." as the system turn (`Untriaged.md`). The mila arm measures MIS as a
  user meets it, and its comparison with `hf` measures the engine and the template together.

### 11.3 The benchmarks

| Benchmark | Dataset | Access | Definition | Fit |
|---|---|---|---|---|
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | open | `mmlu_pro`: zero-shot `multiple_choice`, scored by `choice`, one epoch | Runs today through MIS's Chat Completions route; no judge, no log-likelihood |
| GSM8K | `openai/gsm8k` | open | `gsm8k`: a prompt template and `generate`, scored by `model_graded_fact`, four epochs reduced by `pass_at_1` | Out: a judge model, and with none named inspect-ai grades with the model under test |
| GPQA | `Idavidrein/gpqa` | gated, approved on request at once | Unread: behind the gate | Small models score near the 25% of chance, and Diamond's 198 questions resolve coarsely; its value is on the larger models |
| HLE | `cais/hle` | gated | Unread; the documentation's example scores with `model_graded_fact` and an external grader | Out on present reading: a judge, and near the floor for every model Mila runs |
| IFEval | `google/IFEval` | open | None: not registered | Stays with `run_arm.py` |

GPQA's terms: "You agree to NOT reveal examples from this dataset in plain text or images online, to reduce the
risk of leakage into foundation model training corpora." Scores may be published; logs, samples and any comparison
report that quotes a question may not, which binds section 10's decision 4 for it. Reading its definition needs a
Hugging Face token from an account that has accepted the terms.

### 11.4 Controls on the Hub path

Enforced by the tools, as section 4's are:

- `run_inspect.py` runs the benchmark's own `eval.yaml`, pinned to the revision it read, at temperature 0 with one
  request in flight and the same reply length on every arm, and records the revision, scorers and epochs beside
  the log. It refuses a benchmark with no `eval.yaml`, one on another framework, one scored by a judge model, and
  a gated one the account cannot read.
- `eval_results.py` writes only the mila arm, never a run made with `--limit`, never over an existing file, and
  links a gated benchmark's logs only when told the link is restricted to people who accepted its terms. Its notes
  name the Mila and inspect-ai versions and the decoding; the weight format is given with `--notes`.
