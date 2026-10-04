# Context Profile

**Status:** Draft, 2026-10-03. Agreed in discussion with Todd the same day. Section 9 decided the same day. Phase 1 in
progress from `+34`.
Admitted to v0.21 the same day for every family in `ModelFamilyParity.md` (Todd): Phases 1 and 2, for every model the
release publishes (`ROADMAP.md`, Mila::AI success criteria).

**Area:** measuring what a model configuration is worth to an agent at each context length it can hold, and what
it costs there. Its uses: the reliable depth at which `Mila::AI` compacts a conversation; choosing between Gemma 4
26B-A4B and Qwen 3.8 27B for agentic work on the 16 GB card; and the measured reason K = V needs before it is built
(`RopeInAttention.md`) -- storing the 26B-A4B's global keys once raises what fits from 80K to 128K, which is worth
discussing only if the profile's reliable depth runs past 80K (Todd, 2026-10-03).

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

16K, 32K, 64K, 128K, and the configuration's largest planned context when that is not one of them (from 16K: Todd,
2026-10-03 -- the depths an agent runs at). A band
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
`Tools/Measurement/LogLikelihoodHarness.h` and `Pg19Books.h`.

### 4.3 Recall at depth

Synthetic and deterministic. A conversation of tool results -- book text framed as tool output in the family's own
turn grammar -- carries records with unique keys and values drawn from a seeded generator, placed at 0, 25, 50 and 75%
of the band and just before the question. The last user turn asks for one value by its key. Two scores per cell:

- **Exact recall:** the greedy reply contains the value. A proportion over trials, with its interval.
- **Answer log-likelihood:** the teacher-forced log-probability of the correct value after the question. Continuous,
  so it separates two formats that both pass, and it needs no generation. Forced through the path the reply takes: the
  question prefilled from the end of the conversation, then each token of the value fed by a decode step. Drafted as
  the difference of two prefill scores, with and without the value; that is exact only when a prefill row does not
  depend on the other rows of its chunk, and FP4's prefill quantizes activations to FP8 with a scale shared across the
  chunk, so the question's own score differed between the two calls -- Gemma 4 12B FP4 read +0.6 nats for answers,
  where a log-probability is at most zero, and Q4_0 exactly 0 (2026-10-03, the first gate run). A prefill-path score
  would need `sequenceLogLikelihoodFrom` to score from a position later than the one it prefills from, a library
  change not made; awaiting Todd.

Several records per conversation, so one prefill serves several questions. Values are random identifiers, not facts,
so nothing is answered from the weights.

**Thinking** is a configuration axis (2026-10-04, Todd). Off, as the first profiles ran: each family's primer closes an
empty reasoning span and the reply has 24 tokens -- an answer from memory in one pass. On (`--thinking`): Gemma's
`<|think|>` leads a system turn of its own and the model opens its thought channel itself, as Chat renders it
(`Mila/Src`'s `Gemma::formatPrompt` always closes the channel, so the tool stops its primer at the model turn); Qwen
at Medium, the checkpoint's default, which adds no instruction -- Max is reported unreliable by its users, and is not
a level profiled. No budget is asked for: each model thinks as long as it chooses, and the tokens it spends are
reported, with a reply limit of 4096 and the network sized beyond the band for it, so a thinking run asks the
thinking-off run's questions of its very conversations. The answer is what follows the reasoning's closing marker;
the answer log-likelihood is not scored, teacher-forcing after free-running reasoning being undefined. A reply that
calls a tool instead of answering stops at the call, as an agent's harness would, and is counted apart -- an attempt
to recover, not a wrong value. A reasoning reply longer than Gemma's sliding ring holds overwrites the positions a
rewind to the end of the conversation needs; the rewind is refused, rightly, and the tool prefills the conversation
again, as a model does when it cannot reuse a prefix, and counts the refills. The first thinking run, Gemma 4 12B at 16K, called `get_record` for every never-planted
key after about 700 tokens of reasoning, and for no planted one.

The conversation (decided 2026-10-03, Todd): one user turn setting a task, then assistant calls reading a document
whose results are the book text, and each record its own tool result, a JSON object of a key and a value. No system
rule and no declared tools (section 9, decision 2): the calls are history the model reads, not a choice it makes. Eight
records at each of the five depths, so forty questions a conversation; each question is a user turn asking for one
record's value, then a model turn with thinking off. Every turn is rendered by the family's protocol in `Mila/Src`
(`formatTurn`, `formatToolResponse`); Llama's tool turn is not there yet (`ModelFamilyParity.md` 3.3), and Llama's
recall waits for it rather than the tool carrying a third copy of the grammar.

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
network builder and the log-likelihood harness move from `Tests/Common` to `Mila/Tools/Measurement`, a header-only
target both the tests and the tool link, unchanged but for their namespace (`Mila::Measurement`) and the PG-19 path,
which takes the data root rather than a test macro. A user-facing benchmark (`Untriaged.md`, "A user cannot measure Mila's rates on their own card") may later
promote its core. What the two share is fit and turn cost, which work at the model level; loss and recall need the
network layer, a corpus and a reference, and stay here. Phase 3 is where they meet: if that benchmark is admitted by
then, the timing core moves once to where it ships and both call it (2026-10-03, Todd). Until then the fit section
records the facts such a benchmark would -- configuration, card, reading, plan -- so their outputs line up.

## 8. Phases

**Phase 1 -- fit, loss by band, recall at depth.** The cheapest set that shows the context and quantization trade.
Gate, written before the first run:

- **The loss arm reproduces G2.** Gemma 4 12B Q4_0, the tool held to 65536 so it builds and chooses books as
  `DISABLED_KvCache_12B_Q4_0_Bf16` does -- the first five PG-19 test books that fill it: every band of every book
  within 0.005 nats a token of that test's `[bands]` output on the same build. (Drafted as books 30312 and 3608, which
  are the first to fill 262144 and are not among that test's five; corrected 2026-10-03, Todd.)
- **The recall arm discriminates.** A configuration with a known long-context loss scores lower than one without it:
  Gemma 4 12B FP4 against Q4_0, the pair G2 separated by +0.69 against +0.03 nats a token, at the deepest band both
  fit on the card (131072 on the RTX 5060 Ti). The test is the answer log-likelihood paired over the same planted
  questions: for each, the sign of FP4's less Q4_0's, differences under 0.001 nats not counted, and FP4 must rank lower
  by a two-sided sign test at p < 0.01. If recall cannot tell them apart, it is measuring nothing the loss arm does
  not. (Drafted at 32K on recall alone; rewritten 2026-10-04 with Todd, after the first run found both configurations
  at the ceiling there -- the results below.)
- **The recall arm is not answered from the weights.** Each question asked with its record removed scores at chance.
  Asked in the same conversation for keys drawn from the same generator and never planted: a value is random, so only
  its record ties it to its key, and a conversation without the record is one that never had it -- with no fill per
  question (2026-10-03, Todd).
- **Determinism.** One configuration run twice gives identical JSON, but for the run's own record -- the reading of
  free memory and wall times -- which the output keeps in a section of its own.

**First run of the gate, 2026-10-03** (`0.21.0-dev+34`, RTX 5060 Ti, working files `D:\Claude\context_profile\gate1`,
`gate2`, `depth`). Three of four pass as drafted; the second fails at 32K and passes as rewritten.

- *Reproduces G2: passes.* Five books at 65536, chunk 1024 as in the test: every band of every book within 4.8e-7
  nats a token of the test's `[bands]` line, which prints six decimals -- the same arithmetic. Gate 1 holds in all
  twenty bands.
- *Discriminates: does not, at 32K.* Twelve conversations, 480 planted questions a configuration. Exact recall: FP4 477,
  Q4_0 479; paired over the same questions, two misses are FP4's alone and none Q4_0's alone. Mean answer
  log-likelihood -0.070 nats (FP4) against -0.037 (Q4_0), carried by four answers below -0.1 against one; both medians
  about -0.0001. Every miss names another record's value from the same conversation -- a confusion, not an invention --
  and both of FP4's own name a record from just before the question. Both configurations are at the ceiling of this
  task at 32K, where G2 separates them by +0.69 against +0.03 nats a token.
- *Not answered from the weights: passes.* Every never-planted key, 96 a configuration: 0 recalled, about 5.9 nats a
  token on the would-be value, and the reply "I do not have a record for ..." in every case read.
- *Determinism: passes.* Gemma 4 12B Q4_0 (every arm) and FP4 (fit and recall), each run twice: identical JSON
  outside the run's own record.

The first run also found the answer score's drafted form wrong (4.3); the discrimination rows above are on the
corrected one.

*Deeper, the same protocol* (`D:\Claude\context_profile\depth`; both fit 131072 at chunk 1024). Paired over the same
480 planted questions a band -- the answer log-likelihood's sign per question (differences under 0.001 nats not
counted), and exact recall's discordant pairs -- two-sided sign tests:

| Band | Exact recall FP4 / Q4_0 | Mean answer log-likelihood FP4 / Q4_0 | FP4 lower / higher, p | Exact, FP4 only / Q4_0 only, p |
|---|---|---|---|---|
| 32768 | 477 / 479 | -0.070 / -0.037 | 20 / 10, 0.099 | 2 / 0, 0.5 |
| 65536 | 476 / 479 | -0.042 / -0.009 | 71 / 28, 1.8e-5 | 3 / 0, 0.25 |
| 131072 | 465 / 473 | -0.253 / -0.091 | 244 / 79, 1e-20 | 14 / 6, 0.12 |

The answer log-likelihood separates the pair from 64K and decisively at 128K; exact recall alone never does at 96
trials a cell, but at 128K FP4's 75% depth falls to 84 of 96 against Q4_0's 92, under section 9's 90% -- so on recall
alone FP4's reliable depth is 65536 and Q4_0's 131072. **Under the rewritten check the gate passes**: at 131072 FP4
ranks lower on 244 questions and higher on 79, p = 1e-20. What the 32K row adds: a recall pass where the arm has not
been shown to separate a known loss is weak evidence, so a profile's shallow bands say less than its deep ones.

Then the first profiles: Gemma 4 26B-A4B Q4_0 (FP8 global cache) and Qwen 3.8 27B FP4 and 2.82-bit, on the RTX 5060
Ti, every band each fits; then the rest of the release's models -- Gemma 4 12B Q4_0, Llama 3.2 3B and Llama 3.1 8B.

**First profiles, 2026-10-03/04** (`0.21.0-dev+34`, RTX 5060 Ti, 15,168 MiB free; JSON and Markdown in
`D:\Claude\context_profile\profiles`, the 26B's bands past the card in `kv`). Recall alone, so each reliable depth is
reported on recall (section 9, decision 5); exact recall at the 50% / 75% depths, of 96, where it is lowest:

| Configuration | Fits (planner's choice) | Recall, deepest bands | Loss, gate 1 | Reliable depth, recall |
|---|---|---|---|---|
| Gemma 4 26B-A4B Q4_0, FP8 global cache | 65536 (55296); 131072 refused, 406 MiB over | 64K 94/92; past the card, 96K 90/89, 128K 78/79 | passes, five books to 64K | 98304 -- past what fits |
| Qwen 3.8 27B 2.82-bit | 74752 (74752) | >= 97% every cell to 74752 | passes but for one UTF-8 book (`Vnext.md`, BPE tokenization) | 74752 |
| Qwen 3.8 27B FP4 | 16384 (11264) | >= 99% to 16K | **fails** from 8K (`Vnext.md`) | 16384, but see loss |

The 26B at 128K keeps the start and end of the conversation (every record recalled) and loses the middle; its misses
name other records' values. Its instruction retention, read on the older arm
(`Gemma.InstructionRetention.Cuda.cpp`, `DISABLED_Fp8GlobalCache_26B_Q4_0_PastTheCard`), is 4 of 6 at 64K, 96K and
128K alike, the same two misses each time -- a four-word reply on book 3608 and replies cut at the 256-token budget.
**Thinking on, first runs, 2026-10-04** (`D:\Claude\context_profile\thinking`, two conversations a band, the
thinking-off runs' own questions). Gemma 4 12B Q4_0: planted records 79 of 80 at 64K and 78 of 80 at 128K, against
80 and 77 thinking off -- no change beyond noise. A correct recall took a median of about 100 thinking tokens; the two
128K misses were confident wrong values after about 97, so reasoning does not recover a confusion the model does not
doubt. Every never-planted key at 128K, and 11 of 16 at 64K, ended in a `get_record` call -- recovery where the memory
is empty -- and 5 at 64K, with one planted record, ran to the 4096-token limit unclosed. The 26B-A4B's run was stopped
(Todd): its 64K band read 38 and 29 of 40 by conversation, unexplained, and its 128K band, built past the card at chunk
64 with a full refill after every long reply, had not finished a conversation in 100 minutes. The tool now writes its
output after every conversation, so a stopped run keeps what it measured.

For K = V (`RopeInAttention.md`): the 26B's reliable depth runs past 80K but not to 128K, so what storing the global
keys once would buy is 96K with headroom, about 87 MB over the card today; the allocation rounding that costs it
about 225 MiB (`BACKLOG.md`, Deployment Planning) is the other way to the same band.

**Phase 2 -- instruction retention and tool-call fidelity**, ported from the existing arms and the families' grammar.
With it, a **recovery** arm (raised by Todd, 2026-10-04: recall measures one unaided lookup, and an agent can check
itself): recall's conversations with `get_record` declared and a question that needs the value for a task, scored on
whether the final answer is right and, apart, whether the model looked the record up again. A model that always
re-fetches scores its diligence rather than its memory, which is as much what an agent needs to know. Its first
evidence is above: thinking, the 12B reached for the tool exactly where its memory had nothing, even undeclared.
Recall's misses at depth argue the other way -- the 26B's at 128K were 30 confident wrong values of 35, which give a
model no reason to check.

**Phase 3 -- turn cost**, after Qwen's prefix reuse lands, so Qwen's cells measure the model rather than the
missing reuse. Until then its cells run and say what they measured.

**Phase 4 -- a llama.cpp column** for loss by band, through `llama_cpp_long_context_loss.py`, on the GGUFs of
`benchmark_comparison.py`'s rows.

## 9. Decisions

Decided 2026-10-03 (Todd), before any profile has run.

1. **Trials:** about 96 per recall cell, which bounds the interval near +-10 points at 50%. They come from many records
   per conversation, several questions asked after one shared prefill, each from the end of the conversation -- a
   rewind on Gemma and Llama, a return to the saved state on Qwen. Priced per family before the first run:

   *Priced 2026-10-03, `+33`, RTX 5060 Ti:* one `generate()` per band through ProfileModel, filled to the band and
   then 32 tokens generated, one run, no warm-up -- a price, not a published rate. Seconds to fill one conversation /
   generation rate at that depth, tokens a second:

   | Configuration | 4K | 8K | 16K | 32K | 64K | 128K |
   |---|---|---|---|---|---|---|
   | Llama 3.2 3B FP4, BF16 cache | 0.33 / 131 | 0.81 / 114 | 2.3 / 90 | 7.1 / 64 | 24.8 / 40 | refused, 18.3 GB |
   | Llama 3.1 8B FP4, BF16 cache | 0.70 / 71 | 1.6 / 65 | 4.3 / 56 | 12.5 / 43 | 40.9 / 30 | refused, 23.2 GB |
   | Gemma 4 12B Q4_0, BF16 cache | 1.3 / 50 | 2.8 / 49 | 6.1 / 49 | 14.2 / 47 | 36.0 / 44 | 102.6 / 40 |
   | Gemma 4 26B-A4B Q4_0, FP8 global cache | 0.91 / 118 | 1.7 / 117 | 3.4 / 114 | 7.8 / 108 | 23.5 / 98 (chunk 512) | refused, 16.33 GB |
   | Qwen 3.8 27B FP4 | 3.1 / 22 | 6.4 / 21 | 14.9 / 21 (chunk 512) | refused, 16.67 GB | -- | -- |
   | Qwen 3.8 27B 2.82-bit | 5.7 / 26 | 11.7 / 25 | 24.3 / 23 | 52.1 / 23 | 118.1 / 21 | refused, 18.97 GB |

   Free memory at every reading: 15,168 MiB. With 12 conversations a band, 8 records at each of the 5 depths (96
   trials a cell) and a question costing about 20 decode steps, every question asked from the end of its conversation,
   the recall arm from 16K costs: 26B-A4B 13 min, Qwen FP4 11 min (16K only), Llama 3B 16 min, Llama 8B 25 min, 12B
   48 min, 2.82-bit 61 min -- about 3 h for all six. Refilling the conversation for every question instead would cost
   about 73 h from 4K. Asking from the end needed three library changes, agreed with Todd the same day and made at
   `+33`: Llama's transformer rewinds and continues a prefill as Gemma's does; Qwen returns to a position whose
   recurrent state it saved (`savePosition`, `ModelFamilyParity.md` 8.3, Q5); and `sequenceLogLikelihoodFrom` scores
   after a cached prefix. A question is a rewind to the end of the conversation, a prefill of the question, then
   greedy decode; an answer's log-likelihood is a second rewind and prefill of the question, then the value's tokens
   forced in by decode steps (4.3 says why it is not the difference of two `sequenceLogLikelihoodFrom` calls, as
   first written). The 4K and 8K cells above are kept as measured.
2. **Each arm builds its own conversation.** A system rule or a set of declared tools inside a recall conversation
   would change what recall measures.
3. **Records are JSON objects**, the form of tool output an agent meets most.
4. **The page and the post follow the first profiles**, written from their data.
5. **Reliable depth** is the deepest band at which exact recall is at least 90% at every planted depth, instruction
   retention at least 90%, and tool-call fidelity at least 90% (well-formed, right tool, exact arguments). Until
   Phase 2 lands for a configuration, its reliable depth is reported on recall alone and says so.
