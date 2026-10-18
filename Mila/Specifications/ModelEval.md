# Model Evaluation

**Status:** Draft, 2026-10-07. Agreed in discussion with Todd the same day: start with BF16 parity on Llama 3.2 3B
over IFEval and GSM8K, then add BFCL, RULER and a llama.cpp arm. The tools for Phases 1 to 4 are built
(`Mila/Tools/Evaluation`, `0.21.0-dev+41` and `+42`) and rehearsed end to end on a stand-in model. 2026-10-10
(`+55`): Phase 1's two arms ran on real weights on CUDA for the first time, a 20-document smoke run per task on one
card, which checks the setup and is not a result. No release admits this work; section 10 holds the decisions that would. Added 2026-10-08: scores
for the Hugging Face Hub's benchmark leaderboards (section 11, Phase 7), built at `0.21.0-dev+43` and rehearsed
against a stand-in server on MMLU-Pro's own definition. Added 2026-10-15 (Todd): both Qwen 3.8 builds are measured
for agentic work before either is published as agentic (section 7, Phase 8).

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
consumer-blind: what it gains for this spec is what MIS needs from the library and any program could call --
`version()` and `cuda_devices()` (`+55`), and for Phase 5 the log-likelihood the sessions do not yet expose.

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
| BFCL, `multi_turn` collection | A task carried across turns: base, missing function, missing parameter, long context; 800 entries | generation | 8 |
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

BFCL's **multi-turn** entries fit the controls of section 4 as they stand. The user turns are written in advance and
reviewed by hand, not produced by a simulator, and an entry passes only if, at the end of every turn, the backend's
state matches the expected state and the calls made contain the expected minimal path; a turn of more than 20 steps
fails. No judge model is involved (Gorilla, "BFCL V3: Multi-Turn & Multi-Step Function Calling", 2024). Its memory
and web-search categories are not used: web search needs the network, and the memory categories run backends
whose behaviour has not been read.

**RULER** is lm-eval's port (`ruler` group, 13 tasks). It generates its documents at run time with the tokenizer
it is given, so every arm sees the same documents. Its bands are set with lm-eval's metadata
(`max_seq_lengths`), and it generates 500 documents per band, laid end to end; `--limit` would take only the
shortest band, so `run_arm.py --per-band N` selects the first N of every band. Defaults set with the tool: bands
of 4K, 8K, 16K and 32K, and 100 documents a band, revisited after the first run. 64K is the edge of the 16 GB card
for the `hf` arm (Llama 3.2 3B's BF16 cache is about 114 KB a token, 7.3 GB at 64K, beside 6.4 GB of weights), and
128K does not fit it; past what the reference holds, a band has no reference, and Mila against itself across
formats is `ContextProfile.md`'s. Four of the 13 tasks fetch text at run time (essays and two QA sets).

**Not in scope:** benchmarks scored by a judge model (MT-Bench, Arena-Hard), since the judge becomes a third engine
in the comparison; agent benchmarks whose user is a simulating model (tau2-bench, whose leaderboard simulates the
user with GPT-4.1), for the same reason; and code benchmarks that execute generated code (HumanEval, MBPP), until
there is a sandbox to run them in.

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

`logprobs` and `echo` on MIS's `/v1/completions`, projecting `sequenceLogLikelihood` -- which the binding's
sessions must expose first, since MIS reaches the library only through them -- then the multiple-choice
tasks of section 5. Also gives every log-likelihood task a token-exact comparison without generation, which is a
sharper parity test than Phase 1's.

### Phase 6 -- Every published model

Llama 3.1 8B, Gemma 4 12B and 26B-A4B, and both Qwen 3.8 builds, on the tasks of Phases 1, 4 and 5. Blocked on
section 10's reference decision for every model whose BF16 weights exceed the card. The Qwen builds' agentic
measurement is Phase 8.

### Phase 7 -- Hub benchmarks

MMLU-Pro through `run_inspect.py` on the `hf` and `mila` arms of Phase 1's model, compared by `compare_arms.py`, and
the mila arm's score written to the model repo by `eval_results.py` (section 11). GPQA follows once its definition
has been read (section 11.3).

**Gate:** a model's Hub result is published only once its card carries the paired comparison against the original
(section 10, decision 5). The comparison against `hf` is read as Phase 1's is, knowing it also measures MIS's prompt template (section 9).

**Built:** `run_inspect.py`, `eval_results.py`, and an inspect-ai reader in `compare_arms.py`. Rehearsed against a
stand-in server on MMLU-Pro's real `eval.yaml`, with stand-in rows in its schema (the container could not fetch
the Hub's data files): two served arms paired and compared, the result file written, and every refusal of section
11.4 exercised.

### Phase 8 -- Qwen 3.8 27B for agentic work

The FP4 build and the 2.82-bit build, the latter re-fitted on calibration text that carries tool calls
(`Qwen3.8.md`, *The 2.82-bit build misnames tools, and the packed path is not why*), each measured as Mila calls a
model agentic. Agentic work needs at least 64K of context (`Qwen3.8.md`, *Agentic deployment on the 16 GB card*):
the 2.82-bit build is measured to 64K, and the FP4 build to the most its plan reaches on the 16 GB card, which is
short of it.

| Measurement | Owner | What it settles |
|---|---|---|
| BFCL `python` collection | this spec | Single-turn calls: names, arguments, parallel calls, knowing when not to call |
| BFCL `multi_turn` collection | this spec | A task carried across turns, scored on the state it leaves |
| RULER, bands to 64K | this spec | Retrieval and tracing at the length agentic work runs at |
| Recall, retention and a tool call at depth, to 64K | `ContextProfile.md` | The reliable depth the `agentic` capability is granted on |
| Claude Code, Codex and Hermes Agent through the inference server | `Apps/Server/docs/harnesses.md` | A real session finishes; for Claude Code, at 64K without compacting every turn |

The first three are arms of this spec: the two quantized builds, the 2.82-bit build paired against the FP4 build,
and each measured teacher-forced against the BF16 reference (decision 1). The last two are about the 16 GB card
and run on it.

The phase runs once, on final builds. The 2.82-bit build's re-fit is accepted or redone on its own fast gates
first -- the name probes and the divergence from BF16 below -- so no benchmark run waits inside that loop.

Three things the calibration change brings into this phase, since the 2.82-bit build is fitted on text of the kind
these benchmarks score:

- The calibration text asks no question BFCL asks, which `Tools/Quantization/calibration_text.py` checks
  against bfcl-eval's own data. RULER generates its documents at run time and `ContextProfile.md` plants its
  own sentences, neither drawn from the calibration sources.
- The twelve-name probe of the misnamed-tools item is joined by a second set of twelve
  (`Tools/Quantization/tool_name_probe.py`), and the calibration text names neither set's tools, so passing
  is not recall of calibration.
- The re-fitted build departs from the BF16 original less than the build it replaces: mean and median KL,
  teacher-forced on the held-out tool and chat conversations. This replaced the perplexity ratio against FP4 at
  16K (Todd, 2026-10-17). Perplexity on served text does not rank the builds -- the original scores worst of all
  -- and a ratio scored on wikitext favours whichever build was fitted on wikitext (`Qwen3.8.md`, after the re-fit).

**Gate:** decision 2's thresholds, which are not set. Until they are, the phase reports each build beside `hf` and
beside the other build, and whether a build is published as agentic is decided against those numbers.

**The 2.82-bit build is an agentic candidate (Todd, 2026-10-18).** It is the only Qwen 3.8 build that holds the 64K
minimum on the 16 GB card, and what has been measured holds up; it is not called agentic, on a card or in its
manifest, until it passes, in order: RULER at 32K and 64K at its defaults; `ContextProfile.md` at 64K with thinking
on -- recall, instruction retention, a tool call at depth; and sustained sessions through MIS at 64K on the 16 GB
card -- one real multi-step task each in Claude Code and Codex, and a set of real-world tasks in Hermes Agent, the
same set for every build so the builds can be compared. The Hermes set is to be written: tasks a user would
give an agent, each with a check that says whether it was done.

**Built for decision 1's teacher-forced measurement (2026-10-16):**
`Tools/Converters/Qwen/qwen38_BF16/hf_qwen_conversation_loss.py` scores held-out conversations under the BF16
checkpoint, layer-streamed, and writes its predictions two ways: the top 32 per position (`--top-k-output`, read by
`DISABLED_DivergenceOnConversations`), and the whole distribution in `llama-perplexity --kl-divergence`'s base format
(`--llama-kl-base`, read by `DISABLED_DivergenceAgainstLlamaBase`), so a GGUF build of the same model is measured
against the same reference on the same positions. Results are in `Qwen3.8.md`, after the 2.82-bit build's re-fit.
Median and 99th-percentile KL are reported beside the mean, which a few near-certain positions carry.

**Built (2026-10-17): the BFCL handler for Qwen 3.8** (`Tools/Evaluation/bfcl_qwen38.py`, registered by
`run_bfcl.py` as `Qwen/Qwen3.8-27B-FC`). BFCL's handlers stop at Qwen3, whose calls are JSON; Qwen 3.8 writes
`<tool_call><function=...>` with every argument as text, so the handler renders with Qwen 3.8's own chat template
at its defaults, and types each argument by the entry's schema as it parses the reply -- BFCL's checker sees only
the stored result, where no schema is in reach. Calls are read from the answer after `</think>` only. It stores the
typed calls beside the reply as written. Section 4's control holds with the template as the prompt format.

**A model is measured at its defaults (Todd, 2026-10-18).** Thinking as the template defaults it -- for Qwen 3.8, on
at `xhigh` -- and sampling at the publisher's recommendation, which MIS applies to every knob a request leaves
unset; `run_bfcl.py` sends none unless `--greedy` asks, which is for two engines on one model. This reverses
thinking off for this phase, agreed 2026-10-17 to keep the runs short and rejected on its first results: a score for
a configuration nobody runs says nothing about the model a user gets. A run that must be short shrinks its sample,
never its configuration. Rehearsed with thinking off before that, on the 4070, and still true of the tooling: every
BFCL reply read, `multi_turn_base` rendering every call and tool result back through the template, RULER's 4K band
running end to end.

**A first sample at the model's defaults (2026-10-18).** BFCL's `python` collection, the first 10 entries a category
(110), and `multi_turn_base`, the first 10, thinking on at `xhigh`, recommended sampling, no seed. FP4 on the 5060 Ti
at 32,768 with an FP8 cache; the 2.82-bit re-fit on the 4070 at 10,240. Paired by entry:

| | Single-turn right | Only this one right | Multi-turn right |
|---|---|---|---|
| FP4 | 85 / 110 | 6 | 4 / 10 |
| 2.82-bit, re-fit | 90 / 110 | 11 (McNemar p ~0.3) | 5 / 10 |
| 2.82-bit, thinking off and greedy, same card, same entries | 93 / 110 | 9 against the defaults' 6 (p ~0.6) | 5 / 10, the same five |

At this size neither the format nor thinking moves BFCL detectably; with thinking on the model calls less where no
tool fits (four of its six gains are irrelevance entries). Both builds miss the same kinds: several calls asked in
one turn (made one, or values that do not match), and about half the multi-turn tasks, ending in the wrong state.
Reasoning is mostly short -- a median of 193 tokens an entry, 835 at the 90th percentile -- with one irrelevance
entry that reasoned for 31,726 tokens until the context ran out. Every miss but one is an answer finished and
wrong; the exception is a multi-turn task whose prompt, carrying its earlier reasoning, outgrew the 4070's 10,240.
A sampled run with no seed cannot be repeated, so a flip here mixes the build with the draw.

**RULER at the model's defaults, timed (2026-10-18).** `run_arm.py --model-defaults --reasoning-end '</think>'`, two
documents a task, both builds on the 5060 Ti: FP4 at 16K (its 32,768 context leaves no room to reason past a 32K
document) took about 21 s a document; the 2.82-bit re-fit at 16K and 32K about 47 s a document across both bands.
Two findings. RULER writes the start of the answer into the assistant's turn, so Qwen 3.8's template closes the
reasoning block in the prompt: on RULER the model answers directly, whatever its default. And the run's scores are
void -- the answer extraction dropped every reply that had no `</think>` of its own, which on RULER is all of them;
fixed the same day (`served_defaults.py` reads whether the prompt left the block open). The raw replies read
correct; the scores need the run again (`Data/Evaluation/qwen38-defaults-ruler-timing-*-scoring-defect`).
At those rates, 13 tasks a band: 10 documents a task costs about 1.7 hours a band at 32K for the re-fit, and 64K,
with attention's longer prefill, more than twice that -- arithmetic on two documents, not a measurement.

**Owed before the long runs: RULER's documents a band for this phase.** Section 5's 100 a band costs the 2.82-bit
build, on one card, roughly two and a half days to 64K, extrapolated from the 4K rehearsal and the 17-second 17K
prefill of the agentic section, so not measured. Both builds are paired on the bands both reach, which is 32K for
FP4 with an FP8 cache, and 64K is the 2.82-bit build's alone.

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
  Meta's (`BACKLOG.md`, *Chat and the inference server each hold code that knows which model they are
  running*). The same task through chat completions against Phase 1's run measures what that costs.
- **Mila's tokenizer.** Mila's BPE against HuggingFace's on every Phase 1 prompt: identical ids, or the first
  difference.
- **MIS's tool path.** BFCL measures tool calling from a prompt it renders itself. A user's agent goes through MIS's
  Responses or Messages route, which renders the declarations and parses the calls in each family's own grammar;
  BFCL's categories driven through that route, against Phase 4's run, measure what the route costs.
- **Sampling.** Phases 1 to 4 are greedy, because they compare two engines on the same weights. A model measured as
  a user runs it samples at its publisher's recommendation (Todd, 2026-10-18; Phase 8 does), which needs repeats and
  a different statistic, and a seed the server does not yet take (`Mila/Issues/Untriaged.md`).

## 10. Open decisions

1. **The reference for models whose BF16 weights exceed the 16 GB card.** Llama 3.1 8B at BF16 is about 16 GB of
   weights, Gemma 4 12B about 24, Qwen 3.8 27B about 54: none runs on the card the project measures on. Options: a
   larger card for the `hf` arm alone (a rented one is enough -- the reference does not need Mila); transformers
   with CPU offload, correct but slow; or Mila FP8 as a proxy reference, which is cheap but makes the anchor a Mila
   build, against principle 2. Qwen 3.8 has the further gap that Mila BF16 does not fit the card either, so the
   card holds no floor for it under any option but the first. **A larger card is not available (Todd,
   2026-10-15).** What this machine holds: 16 and 12 GB on its two cards and 32 GB of system memory, against
   51.8 GB of BF16 Qwen 3.8 weights, so an offloaded transformers reference streams part of every forward pass from
   disk. That rate suits teacher-forced work -- one forward over a whole sequence, as KL and top-1 against BF16
   need -- and not a generated benchmark of thousands of replies. Unmeasured: transformers' offload of this
   model at all, and its seconds a forward. **Decided for Qwen 3.8 (Todd, 2026-10-15):** each build is measured
   against the offloaded BF16 reference teacher-forced -- KL and top-1 over chat and tool-call text -- and Phase
   8's benchmarks pair the 2.82-bit build against the FP4 build, which the first measurement ties back to BF16.
   The card says its benchmark scores are against the FP4 build. No floor exists for this model on this machine,
   so its flips are read against Phase 1's. Held for later: BF16 greedy replies for a sample of entries, made by
   the reference verifying the FP4 build's reply and generating only where they part, which gives the reference's
   own greedy reply at a few forwards each.
2. **The thresholds a published format must meet**, as a difference against the reference beyond the floor. Not
   set until Phases 1 and 2 have produced a floor and a cost to set them against.
3. **Admission. Decided 2026-10-10 (Todd):** evaluation is part of each family's work in v0.21.0, every family
   (`ROADMAP.md`, the Gemma 4, Qwen 3.8 and Llama 3 criteria; `BACKLOG.md`, *Nobody can say whether a published
   model scores like its original on standard benchmarks*). Phase 1 on Llama 3.2 3B sets the floor the others are
   read against.
4. **Where the run data lives.** `Data/Evaluation/` is gitignored. A number on a card cites a run that should be
   retrievable later -- a release asset, or a committed JSON summary beside the card. A gated benchmark's run (GPQA,
   HLE) cannot be public at all: its logs quote the questions its terms forbid publishing (section 11.3).
5. **Whether Mila publishes Hub results, and for which models. Decided 2026-10-10 (Todd): only where the card's
   paired comparison already stands beside it.** Every result on the leaderboards read so far is self-reported
   under its author's own settings (section 11.2), which is the comparison principle 1 refuses; a Mila result is
   reproducible and states its settings, but it sits in a table that is not. So the controlled number is always on
   the card next to the uncontrolled one, and a model whose card has no comparison yet has no Hub result either.
   Rejected: every model, with the notes carrying the settings; and none at all.
6. **Verification.** A verified result needs the run made in HF Jobs (section 11.1), which would need MIS and its
   weights in a Hugging Face job, and Hugging Face's word that a token is issued for an engine it does not host.
   Not pursued; decision 5 no longer blocks it.

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
  MIS then adds "You are a helpful assistant." as the system turn (`BACKLOG.md`, *Chat and the inference server
  each hold code that knows which model they are running*). The mila arm measures MIS as a user meets it, and its comparison with `hf` measures the engine and the template together.

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

## 12. Which configuration to run: the choice matrix

**Status:** captured 2026-10-18 from Todd's question -- when is a smaller model of a family the better choice for a
user than a larger one quantized hard, for a given use case and card? Nothing has been measured for it as such.

A **configuration** is a model, its weight format and its cache format, on one card. For each use case and card
tier the matrix names the cheapest configuration that meets the use case's thresholds -- or says that none does,
which is an answer too. It is the question a user brings to a model card and the website, and today only one card
answers it, for one pair: the 2.82-bit Qwen 3.8 27B at 12 GB, its FP4 build at 16.

### 12.1 What decides it, in order

1. **Fit at the use case's minimum context.** The planner answers this exactly, with no measurement
   (`Deployment.md`): weights and cache compete for one card, so a larger model at fewer bits often fails here
   before quality is asked. A reasoning model's minimum includes its reasoning: one Qwen 3.8 reply at its default
   effort ran to 31,726 tokens (2026-10-18), and its card recommends allowing far more.
2. **Quality at that context**, measured at the model's defaults: section 7's benchmarks against the original,
   `ContextProfile.md`'s reliable depth, and the teacher-forced divergence `Quantization.md` owns as support. The
   abilities most fragile under quantization -- exact copying (tool names, identifiers) and long-context fidelity --
   are the ones agentic work needs, so a short-prompt benchmark alone can rank wrongly.
3. **Rate.** Decode is bound by the weight bytes read per token, so a mixture of experts is priced by its active
   parameters: Gemma 4 26B-A4B reads about 4B a token whatever its total.

### 12.2 What is known

- **Near 4 bits and above, the larger model usually wins.** At a fixed number of weight bits, 4-bit is close to
  optimal, and a larger model at 4 bits beats a smaller one at 16 (Dettmers and Zettlemoyer, "The case for 4-bit
  precision: k-bit Inference Scaling Laws", arXiv 2212.09720, 2022); larger quantized models can outperform smaller
  ones (Jin et al., "A Comprehensive Evaluation of Quantization Strategies for Large Language Models", arXiv
  2402.16775, 2024). Both are post-training results.
- **Trained low-bit models may move the line.** Ternary, 2- and 3-bit models hold the size-accuracy frontier as
  well as or better than 4-bit, in a setting that fine-tunes the low-bit model (Liu et al., "ParetoQ", arXiv
  2502.02631, revised 2025; read from the abstract, whose training details need the full paper). So the answer
  depends on how a build was made, not only its bit width.
- **A format the model was trained for is a different case.** Gemma 4's Q4_0 is Google's quantization-aware build;
  Mila's post-training FP4 is the fallback where no such build exists.
- **Below 4 bits it depends on the method.** Qwen 3.8 27B against BF16 on held-out tool conversations, the share of
  positions whose top token is the original's: FP4 84.4%, 2.82-bit re-fit 78.9%, the build it replaced 73.1%
  (`Qwen3.8.md`). The replaced build also misnamed tools.
- **Low bits cost most at length, and a short-context ranking misses it.** At short context, FP4 hurts the MLP
  projections most and attention least (Cim, Topcu and Kandemir, "Diagnosing FP4 inference", arXiv 2603.08747,
  Qwen2.5, no long-context evaluation). Mila's Gemma 4 12B measurement agrees over the first 8K, and reverses with
  length: attention at FP4 costs +0.11 at 8K and +0.47 at 32K while the FFN stays near +0.1 (`ModelFamilyParity.md`
  8.2, G2).
- **On the 16 GB card:** Qwen 3.8 27B FP4 holds 32,768 with an FP8 cache, the 2.82-bit build 74,752; the 26B-A4B
  at Q4_0 left Claude Code about 7,000 tokens of working room. Only the 2.82-bit Qwen reaches the 64K agentic
  minimum among models of 26B or more (`TieredKvCache.md`).
- **Not measured:** whether any smaller model -- Gemma 4 12B, Llama 3.1 8B -- is good enough as an agent at 64K, or
  any smaller model at higher precision against a larger one at low bits, on the same tasks.

### 12.3 The axes

Card tiers by memory, with the 16 GB card the reference. Use cases, each with a minimum context and its own
thresholds: chat, tool calling and agentic work (64K, Todd 2026-10-15), and long-document work (retrieval at
depth). Candidates: every configuration of every family the planner says fits, published or not.

### 12.4 Open

1. The use cases and their minimum contexts beyond agentic work.
2. The thresholds per use case -- decision 2 in section 10, per use case rather than per format.
3. Which smaller models enter the candidate set, and in which formats.
4. Where the answer is published: each card's "choose this model when", and a page on the website.
5. Whether a configuration that misses a threshold by less than the benchmark resolves counts as meeting it.
