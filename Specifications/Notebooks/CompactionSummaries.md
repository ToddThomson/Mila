# Compaction summaries: length, cost and thinking

Evidence for `Agents/Compaction.md` section 9, measured 2026-11-03 before any compaction code. Scripts and logs are in
`D:\Claude\compaction` (`measure.py`, `probe2.py`, `serve.ps1`).

## Setup

- RTX 5060 Ti, through the inference server (`Apps/Server`, OpenAI chat), which the agent's prompts render the same as.
- Gemma 4 12B Q4_0, context 40960, BF16 KV cache, thinking off unless stated, the template's default. The server turns
  thinking on when a request names nothing, so each request named it.
- Qwen 3.8 27B FP4, context 45056, FP8 KV cache, the template's default: thinking on, `xhigh`.
- Publisher's sampling. One real read-only tool session per model, over this repository, saved between rounds at two
  depths. Summary budget 8192 tokens.
- The instruction names what to keep and says not to call a tool. Three requirements are stated once, in the first
  message: a date, what to call the reader, and a thing never to propose.

**Limits.** One session per model: Gemma 12 rounds, Qwen 5, its reads made in parallel in one response. Small samples. No
second compaction. Summaries kept their requirements by pattern match on all but the empty one, and three were read by
hand: whether a model can carry on from one is the mechanism's test, not this one. The server sends a reply whole
(`Untriaged.md`), so prefill is timed by asking for one token.

## Reading the summary prompt

Warmed on the prompt of the last round generated, then the whole conversation plus the instruction, one token asked.

| Model, conversation | Tools kept | No tools | Skipped, tools kept |
|---|---|---|---|
| Gemma, 20K | 2.62 s | 7.94 s | 14,600 of 20,439 |
| Gemma, 33K | 3.19 s | 14.50 s | 27,403 of 33,246 |
| Qwen, 25K | 12.18 s | 25.28 s | 14,766 of 25,351 |
| Qwen, 32K | 9.31 s | 34.33 s | 25,266 of 32,419 |

**The first probe was wrong and is not used.** It warmed on the whole conversation, whose prompt ends in the assistant
primer. Gemma read 0.09 to 0.13 s, which leaves out the latest results; Qwen could not extend it at all and read from
zero either way. With tools kept, the cost is the latest reply, its results and the instruction.

## How long a summary runs

| Model, setting | Runs | Generated tokens | Finished under 8192 | Whole request |
|---|---|---|---|---|
| Gemma, thinking off | 8 (2 no tools) | 609 to 853 | 8 | 12 to 18 s |
| Gemma, thinking on | 4 | 1,680 to 2,793 | 4 | 34 to 58 s |
| Qwen, `xhigh` | 6 (2 no tools) | 7,202 to 8,192 | 2 | 327 to 367 s |
| Qwen, `xhigh`, asks for brief reasoning | 4 | 2,950 to 7,069 | 4 | 129 to 296 s |
| Qwen, `medium` | 4 | 1,543 to 7,589 | 4 | 87 to 342 s |
| Qwen, thinking off | 4 | 2,180 to 4,581 | 4 | 123 to 220 s |

The whole request includes its prefill. Each run was warmed on its whole conversation first, so Gemma's requests read almost
nothing, and Qwen's read from zero (25 to 35 s). The brief row was warmed on the last round's prompt, the real state, so it
includes 9 to 12 s.

Qwen's `xhigh` ran out of budget in 4 of 6, one with no answer at all. A finished run at `xhigh` was about 60% reasoning by
characters. No summary called a tool: 26 ran with tools offered, 2 Gemma and 2 Qwen ran without.

Gemma's length did not grow between the two depths; Qwen's varied too much run to run to say.

## Carrying on from a summary

Measured 2026-11-03 once the mechanism was built (`real_run.py`). Mila Desktop's session in Build & Code over a copy of
`Lib/Src/Agents`, context 16,384, `summary_tokens` 6,000, thinking off for the summary. The first message states a rule
(begin every reply with "Maintainer:") and a fact (a review date); later messages ask for reads and short explanations, and
the last asks for the date and the first function discussed, with nothing read.

| Model, instruction | Compactions | Replies keeping the rule | Date and first function |
|---|---|---|---|
| Gemma 4 12B Q4_0, short session, first wording | 2 | all | kept |
| Gemma 4 12B Q4_0, long session, first wording | 4 | 3 of 5 | not reached |
| Gemma 4 12B Q4_0, long session, current wording | 6 | 6 of 6 | kept |
| Qwen 3.8 27B FP4, first wording | 3 | 3 of 3 | kept |
| Qwen 3.8 27B FP4, current wording | 6, then 4 failures | all 6 replies until the dead end, then none | no reply |
| Gemma 4 12B Q4_0, long session, as built (`real-gemma3`) | 4 | 5 of 5 | date kept, function lost |
| Qwen 3.8 27B FP4, every message, as built (`real-qwen4`) | 7, then 1 failure | 4 of 4, and every reply | kept |
| Llama 3.1 8B FP4 | 0, trigger not reached | - | - |

**The failure.** At the fourth compaction Gemma's summary opened with "Maintainer:" rather than recording the rule, so the rule
was obeyed once and then gone. Asking for instructions to be quoted as instructions, and not applied to the summary itself, cured
it in the one re-run made. One session, one rule, one fact: not a retention rate.

**Llama** never reached the trigger. It gave short vague replies and named the wrong first function, which may be the filed
made-up result defect; neither was traced. One earlier Llama run wrote a runaway reply too long to read, which led to the
cut of an oversized turn; it did not recur.

**Clearing reasoning alone** is what the small steps are: Gemma 12,347 to 12,180 and Qwen 12,740 to 11,353 took no time and wrote
no summary, and the next boundary, seconds later, wrote one (14,902 to 3,922).

**Qwen on the current wording dead-ended,** at step 7 of 10 (`real-qwen2`, summary budget 4,096, a quarter of this context). Six
compactions held the rule, then:

1. Between rounds at 15,383, the tail did not fit beside the budget, so it was left out. The summary prompt then ended in the
   response's own message, unanswered, followed by the instruction. The model called a tool, in the summary's first failure of that
   kind. The prompt's shape was the first theory; the replay below disproves it.
2. The round went on, generated 1,002 tokens of reasoning, reached the limit and ended. That reply stayed, and the conversation
   now held 16,418 tokens in a context of 16,384.
3. Every later message tried a summary, ran out its 4,096 tokens in about 170 s, failed, and was refused. Three in three.

So 4,096 sits below the longest thinking-off summary measured here (4,581), and a failed summary over the limit leaves nothing to
go on with.

**As built, Qwen reached the end,** 2026-11-04, all eleven messages in 20 minutes, summary budget 4,096. Five of its seven compactions wrote
a summary (49 to 149 s) and two only cleared reasoning (12,377 to 11,894 and 12,733 to 11,694, no time). The summaries left 5,764 to 8,360
tokens, and the last summary is 13,048 characters. At the last message the summary ran out its 4,096 tokens in 167 s and failed; the
conversation stayed at 12,565, which fit, so the message was answered with no refusal. One run. It does not show what happens if failures
go on: each message would spend about as long on one more try, and the turns dropped unread are still unmeasured in a real session.

**Gemma and Qwen lose different things.** Gemma rewrites (`real-gemma3`): the last summary is 3,991 characters and holds `Agent.ixx`
and not `respond`, so asked for the first function it named the file. Qwen's kept `respond`, with the length that risks its budget.
One run each.

## Replaying the dead end

Measured 2026-11-03 and 04 (`replay.py`, `replay.jsonl`). The saved Qwen conversation through the server, context 24,576 so no
budget cut anything, thinking off, Mila Desktop's Build & Code prompt and six tools, a token count within 36 of the real one at
15,383. Asked for at most 8,192. A named length is "Keep it under 2,000 words" unless the arm says. A cut at 4,096 is read from the token counts.

| Arm | The state | Tools offered | Length named | Runs | Called a tool | Finished, tokens | Under 4,096 |
|---|---|---|---|---|---|---|---|
| A | where it called a tool, as built | yes | no | 5 | 1 | 2,948 to 3,170 | 4 of 4 |
| B | A, the message left out | yes | no | 5 | 3 | 3,523 to 4,142 | 1 of 2 |
| C | B | yes | yes | 5 | 1 | 2,948 to 3,386 | 4 of 4 |
| D | the dead state, oldest dropped | yes | no | 3 | 1 | 4,401 to 4,871 | 0 of 2 |
| E | D | yes | yes | 3 | 0 | 2,948 to 3,868 | 3 of 3 |
| F | A | no | yes | 4 | 0 | 3,047 to 3,206 | 4 of 4 |
| G | D | no | yes | 3 | 0 | 3,711 to 3,900 | 3 of 3 |
| H | A, 1,638 words, the fraction built | yes | yes | 4 | 2 | 3,042 to 3,109 | 2 of 2 |
| I | A, 1,638 words, asked to rewrite and not copy | yes | yes | 4 | 0 | 3,106 to 3,352 | 4 of 4 |
| J | D, as I | yes | yes | 3 | 0 | 2,948 to 4,135 | 2 of 3 |

Every finished summary kept the rule and the date, by pattern match. Words ran 1.8 to 2.04 tokens each. A call took about 15 s and a
finished summary 129 to 205 s.

**That pattern match proves little, because the summaries copy.** The conversation opens with an earlier summary of 11,945
characters (1,575 words, 2,948 tokens). Of 31 finished summaries, all open with at least 397 characters of it word for word, 30 with
9,672 or more, and 4 are the whole of it and nothing else: arms A, C, E and J, each byte-identical, which is how this was found.
Those 4 lost everything since it, and kept the rule and the date. Each compaction therefore adds to the last, which is what
pushed the summary toward its budget. Asking for a rewrite did not stop it (arms I and J). Whether Gemma does this was not looked at.

- **The unanswered message was not the cause.** Calls were commoner with it out, in five runs each.
- **Tools were called only under this prompt.** None of 26 above called one, with no Mila Desktop system prompt. That the prompt is
  why is a guess, not tested.
- **Naming a length is what fit the budget:** 22 of 23 under 4,096 with one, 5 of 8 without, though models run over what is asked.
- **Not measured:** whether a summary that copies this much loses what matters over many compactions, and the last resort, which
  nothing here reaches. Small samples, one conversation, one model.

## A length named, with thinking on

Measured 2026-11-04 (`thinking_length.py`), Gemma 4 12B Q4_0 through the server, asked for at most 8,192, three runs each. "Named" is
under 2,400 words. A test failing on its budget raised the question, and the length is not the cause.

| Conversation, thinking | No length, tokens | Named, tokens |
|---|---|---|
| A table of records, about 1,000 tokens, on | 1,672 to 7,140 | 6,938 to 8,192 (one cut) |
| The replayed conversation, about 11,900 tokens, on | 958 to 8,035 | 2,762 to 3,347 |
| The replayed conversation, off | - | 1,073 and 1,154, two runs |
| Four short turns, on, four runs, 1,200 words | 1,080 to 5,042 | 1,448 to 4,586 |

Thinking on varies too much to budget for, with or without a length, and on a table of records it mostly goes over the table again.
Chat and Mila Desktop turn it off for a summary.

## Outside

Others met the same failure on Qwen 3.8. Qwen Code turns thinking off on its summarizer and caps its output
(`chatCompressionService.ts`); Osaurus issue 3036 reports every token going to reasoning and an empty answer, fixed by
`enable_thinking: false`; pi-prefix-cache-compaction keeps thinking and the cache, and asks for brief reasoning. Qwen Code
issue 7332: a thinking-only model refuses `enable_thinking: false`, so it cannot be sent blindly. None measures whether
thinking off lowers a summary's quality, and neither does this.
