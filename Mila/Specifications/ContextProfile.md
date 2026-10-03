# Context Profile

**Status:** Draft, 2026-10-03. Agreed in discussion with Todd the same day; no code. Section 9's decisions are open.

**Area:** measuring what a model configuration is worth to an agent at each context length it can hold, and what
it costs there. The first use is choosing between Gemma 4 26B-A4B and Qwen 3.8 27B for agentic work on the 16 GB
card, and the long-context quality gate that K = V's 128K needs (`RopeInAttention.md`).

---

## 1. The question

A **configuration** is a model, a weight format, a KV-cache format and a card. For one configuration, the profile
answers, at each context length it fits:

- how much worse it predicts text than a reference, and whether it still uses what is far back;
- whether it can find an exact fact placed at a given depth;
- whether it still follows an instruction given before that depth;
- whether it still emits a correct tool call there;
- what a turn costs there, and whether it fits at all.

Comparing two profiles of one model isolates a format (FP4 against 2.82-bit, an FP8 cache against BF16). Comparing
two models at equal fit answers which is the better agent on that card.

**The headline is the reliable depth:** the deepest band at which the configuration still recalls planted facts and
still obeys its system instructions, against the thresholds the profile states. A configuration's useful context is
the lower of what fits and its reliable depth, and the two can differ by a factor of two: Llama 3.1 8B fits 131072 on
the 16 GB card and drops its system rule at 65536 (`Untriaged.md`, HuggingFace reproducing it). The reliable depth is
also where a long session compacts (`ROADMAP.md`, Mila::AI), so the profile is what makes that trigger a measurement
rather than a guess.

## 2. Why one tool

The pieces exist, scattered across gates written one at a time:

- **Loss by band** (`ModelFamilyParity.md` 8.2, G2; `Gemma.LogLikelihood.Cuda.cpp`): PG-19 books in a model turn,
  the prefix of 2L less the prefix of L, gate 1 against the last 1024 tokens alone. It found Gemma 4 12B FP4's cost
  over BF16 growing from +0.19 to +0.69 nats a token by 32K, carried by 4-bit query and key projections.
- **Instruction retention** (`Llama.InstructionRetention.Cuda.cpp`, `Gemma.InstructionRetention.Cuda.cpp`): a system
  rule, a book, a reply. It found Llama 3.1 8B losing the rule at 64K, the model's own failure.
- **Fit** (the planner, `Deployment.md`): exact, and it allocates nothing.
- **Rates** (`ProfileModel`, `benchmark_comparison.py`).

Each answers one question for one family at one moment. An agent's question needs all of them, per configuration,
on the same bands, in one output.

**Averages hide the failure an agent suffers.** Quantization error compounds with depth through attention, and it
shows first as a missed fact or a dropped instruction rather than a uniformly higher loss -- the warning the KV-cache
survey makes about eviction (`Quantization.md` Part III). So the profile carries targeted probes beside the loss.

## 3. Bands

4K, 8K, 16K, 32K, 64K, 128K, and the configuration's largest planned context when that is not one of them. A band
the planner refuses is reported as refused, with the planner's bytes, and its other arms are skipped.

## 4. Arms

### 4.1 Fit

The planner's answer at each band: fits or refused, the prefill chunk, the footprint split into weights, state and
scratch, against one reading of free memory. Exact by `PlanEqualsBuild`. Costs a header read and a graph
construction per band; nothing is allocated.

### 4.2 Loss by band

G2's protocol unchanged: books from the PG-19 test split inside a model turn, scored at prefixes doubling to the
largest fitting band, a band being one prefix less the one before. Reported per band: nats a token, the cost over the
reference (section 5), and gate 1 (whole book against window-only). The harness is
`Tests/Common/LogLikelihoodHarness.h` and `Pg19Books.h`.

### 4.3 Recall at depth

Synthetic and deterministic. A conversation of tool results -- book text framed as tool output in the family's own
turn grammar -- carries records with unique keys and values drawn from a seeded generator, placed at 0, 25, 50 and 75%
of the band and just before the question. The last user turn asks for one value by its key. Two scores per cell:

- **Exact recall:** the greedy reply contains the value. A proportion over trials, with its interval.
- **Answer log-likelihood:** the teacher-forced log-probability of the correct value after the question. Continuous,
  so it separates two formats that both pass, and it needs no generation.

Several records per conversation, so one prefill serves several questions. Values are random identifiers, not facts,
so nothing is answered from the weights.

### 4.4 Instruction retention

The existing arm, generalized: a system rule checkable on the reply (a fixed first word, a reply length, a format),
the band filled with tool output, then an ordinary request. Reported as the proportion of replies that keep the
rule.

### 4.5 Tool-call fidelity

The band filled with tool history, then a request that needs a call to one of several declared tools. Scored on a
well-formed call, the right tool and exact arguments, parsed by the family's grammar in `Mila/Src`. Gemma and Qwen
have their grammar there; Llama's is in Chat and MIS (`ModelFamilyParity.md` 3.3), so Llama waits for it.

### 4.6 Turn cost

What an agent feels at depth D: the time to absorb a tool result of 1K and of 2K tokens appended to a conversation of
D tokens, then the generation rate at D. With prefix reuse a turn costs its new tokens; without it, the whole
conversation, and the arm reports which happened. ProfileModel's timing core: salted prompts, a warm-up, three runs,
decode timed from first token to last.

## 5. The reference

A cost needs something to be a cost against. In order of preference:

1. **BF16 in Mila**, where it fits the card.
2. **BF16 in HuggingFace, layer-streamed** (`hf_gemma_layer_stream.py`; Qwen's equivalent), for the arms that score
   text: loss by band, and recall's answer log-likelihood. Slow, but it exists for both 26-27B models, and it is the
   reference G2 used.
3. **The highest-precision configuration that fits**, named as the reference in the output, when neither of the first
   two is affordable for a band.

Exact recall, retention and tool-call fidelity are absolute, and need no reference.

## 6. Output

One JSON file per configuration -- the configuration, the build, the card, the reading of free memory, and every arm
at every band -- and a Markdown rendering of it. A second command compares profiles into one table per arm and the
curves a page or a post draws: cost over the reference against context, recall against depth, turn time against
depth. Data first, so a rerun is the page's update and no number is copied by hand, as `benchmark_comparison.py` does
for the rates page.

## 7. Where it lives

`Mila/Tools/ContextProfile`, developer tooling: it does not ship (`Tools/README.md`). The PG-19 reader, the measured
network builder and the log-likelihood harness move from `Tests/Common` to a library both the tests and the tool link,
unchanged. A user-facing benchmark (`Untriaged.md`, "A user cannot measure Mila's rates on their own card") may later
promote its core.

## 8. Phases

**Phase 1 -- fit, loss by band, recall at depth.** The cheapest set that shows the context and quantization trade.
Gate, written before the first run:

- **The loss arm reproduces G2.** Gemma 4 12B Q4_0, books 30312 and 3608: every band within 0.005 nats a token of
  `DISABLED_KvCache_12B_Q4_0_Bf16`'s output on the same build.
- **The recall arm discriminates.** A configuration with a known long-context loss scores lower than one without it:
  Gemma 4 12B FP4 against Q4_0 at 32K, the pair G2 separated by +0.69 against +0.03 nats a token. If recall cannot
  tell them apart, it is measuring nothing the loss arm does not.
- **The recall arm is not answered from the weights.** Each question asked with its record removed scores at chance.
- **Determinism.** One configuration run twice gives identical JSON.

Then the first profiles: Gemma 4 26B-A4B Q4_0 (FP8 global cache) and Qwen 3.8 27B FP4 and 2.82-bit, on the RTX 5060
Ti, every band each fits.

**Phase 2 -- instruction retention and tool-call fidelity**, ported from the existing arms and the families' grammar.

**Phase 3 -- turn cost**, after Qwen's prefix reuse lands, so Qwen's cells measure the model rather than the
missing reuse. Until then its cells run and say what they measured.

**Phase 4 -- a llama.cpp column** for loss by band, through `llama_cpp_long_context_loss.py`, on the GGUFs of
`benchmark_comparison.py`'s rows.

## 9. Open decisions

1. Records per conversation and trials per cell for recall at depth: enough for an interval of about +-10 points at
   the 50% depth, priced in prefill time before the first run.
2. Whether a band's arms share one prefill where the books allow it, or each arm builds its own conversation.
3. The record format: key-value lines, JSON objects, or table rows. All three are tool output an agent meets; one is
   enough for Phase 1.
4. When the profile becomes a published page and a blog post: after the first profiles, written from their data.
5. The thresholds that define the reliable depth -- for example exact recall of at least 90% at every planted depth
   and retention of at least 90% -- set before the first profile is run, so no profile chooses its own bar.
