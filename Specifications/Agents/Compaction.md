# Compaction

How an agent carries a conversation past the point where its context, or its model's reliable depth, would end it:
what triggers it, what the model is asked, what the conversation holds afterwards, and what it costs on each family.

**Status:** agreed 2026-11-03 and 04 (Todd, decisions in section 9). The mechanism is built in `Lib/Src/Agents/Agent.ixx`; the depths
that set its trigger are not yet measured (section 8). Carries `BACKLOG.md`, *A conversation that fills its context stops*,
`ROADMAP.md`, *A long session continues*, and the seam of `AgentTasks.md` 7.

---

## 1. What happened before it

Read from `Lib/Src/Agents/Agent.ixx` on 2026-11-03, and still what happens with no policy set:

- **A message that does not fit** returns `ContextOverflow` and is dropped; rounds is 0, and Chat and Mila Desktop tell
  the person to start again or choose a longer context.
- **A result that does not fit** is cut to what fits, or refused (`addResult`).
- **A round that reaches the limit** ends the response. Its reply stays in the conversation, so the next message usually
  fails too.

Nothing shortens the conversation. An agent's session may also be changed by the program between responses, through
`Session::turns()`.

## 2. The shape

Off unless the program asks (decision 4). Chat and Mila Desktop ask.

```cpp
session.withCompaction( {
    .trigger_tokens = 40'000,        // absent: the context limit less summary_tokens
    .summary_tokens = 2'000,         // the most the summary may generate, reasoning included
    .kept = { "The goal: ...", },    // copied into the new history word for word
    .instruction = "...",            // absent: the library's
    .template_variables = { { "enable_thinking", "false" } },  // for the summary alone
    .summarizer = {},                // absent: this model writes it
} );

agent.compact( on_event, stop );     // at any time between responses, whatever the trigger
```

`CompactionPolicy` belongs to the session, since it outlives a model, and holds no number about any one model. `kept` is
the list `AgentTasks.md` 7 asks for: a task fills it with its goal and plan, and the program may change it between
responses. `template_variables` are laid over the session's for the summary alone; one this model's template does not
accept is left out, as a session's are, since a thinking-only model refuses the switch. The system prompt needs no entry: it is not in the conversation, and every prompt renders it.

`compact` is one more method on the agent, so `Direction.md` 4.2 rule 1's list grows by one.

## 3. When it runs

At every point where the agent already holds a whole prompt, and only when a policy is set:

| Where | Check | The tail kept as it stands |
|---|---|---|
| Before a response, the message joined | the prompt reaches the trigger, or does not fit | the message |
| Between rounds, every result in | the next prompt reaches the trigger | the latest reply with calls, and its results |
| In `addResult`, before anything is cut | the next prompt does not fit | that reply, and the results so far |

**The trigger is `min( trigger_tokens, context limit - summary_tokens )`.** So a depth past what this plan's context
holds cannot leave a compaction with no room, an absent depth is "the context limit alone" (decision 3), and a session
moved to another model needs no new number.

A message too long for the context even alone is refused as today, and compacts nothing.

A round that reaches the limit part way through a reply still ends the response. There is no boundary inside a
generation. The next message compacts. A trigger well below the limit is what makes this rare, and it is a limit of
this design.

## 4. What it does

1. **Find what can be replaced:** every turn before the tail. If that is nothing, or only an earlier summary, nothing is
   done: this is what stops a session whose tail alone holds the context from compacting every round. The model then
   runs to the limit and ends as today. (`Claude Code` has the same failure and stops after a few attempts, per its
   documentation, *When context fills up*.)
2. **Reasoning, if that alone is enough** (section 5.2). Clear `Turn::reasoning` from the replaced turns when the prompt
   then falls below the trigger, and stop there.
3. **Make room.** The summary prompt is the whole conversation, the tail included, then a user turn holding the
   instruction: the model reads what it holds, and the prompt extends the state (5.1). A message not yet answered is the
   exception: it is carried word for word (step 6) and left out, since a summary that reads it answers it. Where that plus
   `summary_tokens` does not fit, take in order: leave out the tail, which stays verbatim anyway; clear the reasoning; drop
   the oldest turns until it fits, a reply with its results, an earlier summary last; and when one turn alone still does not
   fit, cut that turn's text to what does. What is dropped or cut reaches no summary, and the event counts what was dropped.
   Leaving out the tail covers a large result arriving near the trigger. Codex drops too when its summary request
   overflows, removing the oldest item and retrying (`codex-rs/core/src/compact.rs`). The cut covers a runaway reply: the real
   Llama run left one reply too long to read, and without it the session could never compact again.
4. **Generate the summary**, at the session's sampling and template variables, with `summary_tokens` as its budget and
   no tool run. Or call the program's `summarizer`. None of it is a reply: no `Token`, `Answer` or `Reasoning` event, and
   nothing joins the conversation. The library's instruction asks for under two fifths of `summary_tokens` in words; a
   program's own instruction has nothing added.
5. **Check it.** Failure is a reply cut by its budget, a reply that makes a call, an empty answer, a reply the earlier summary
   already holds whole, or stop. **A reply that makes a call or adds nothing is written again once, at once, with no tools
   offered,** and only a second such reply is a failure. Failure leaves
   the conversation exactly as it was, with the reasoning restored. Hermes also refuses a truncated summary, since a
   partial one loses context without saying so (`agent/context_compressor.py`).
6. **Replace.** The conversation becomes one user turn and the tail:

   ```
   user:      <the summary, under a fixed heading>
              <each entry of `kept`, word for word>
              <the message this response answers>
   assistant: the latest reply with calls        (between rounds only)
   tool:      its results
   ```

   The reply keeps its reasoning: it is the turn in progress, and clearing it is the failure `ModelFamilyParity.md` G3
   closed. The in-flight continuation is dropped, and the next round renders the conversation again.

7. **Report it** (section 6), and go on. If the prompt still does not fit, what happens today happens: the result is cut,
   or the message refused.

**One failed attempt per response.** A summary that failed costs its prefill and generation, so the boundaries after it
do not try again; the response runs to the limit and ends as today, and the next response may try once more.
`compact()` always tries.

**The last resort, before a response only.** When a summary failed and the message does not fit, the oldest turns are
dropped unread until the prompt is below the trigger, so the next message does not pay for another failed summary. A stop
is not a failure, and drops nothing. An earlier summary stays. What is left opens with the summary's heading, or that earlier
summary, then a sentence saying turns are gone, then `kept`. That text joins the first turn left, which is always a user
turn, so no template is given two user turns in a row; when nothing is left, it joins the message as a summary's turn does.
`Compacted` reports it with no text, the dropped turns in `replaced_turns` and their count in `dropped_turns`, and
applications say so. `compact()` never does this: with no message waiting there is nothing to save.

**What stays:** the tools allowed for the conversation, since it is the same conversation (`Session::newConversation`
is what forgets them); the sampling; the template variables.

**Limits:** the response's `time` keeps running, as for a tool call, since it is time the person waits. `reply_tokens`
does not count a summary, since the program set it for what the model says to them, and `summary_tokens` bounds this.

## 5. What it costs

### 5.1 Each family, read from `generate`

| | Gemma, Llama | Qwen |
|---|---|---|
| Reuse | Longest common prefix (`kv_token_history_`) | Only a prompt that extends what the state holds, or the prompt saved last |
| The summary prompt, whole, tools kept | Prefills from the first token that renders differently: the latest reply, its results, then the instruction | Prefills from the saved prompt's end: the latest reply, its results and the instruction |
| The summary prompt, tail left out | Prefills the instruction | **Prefills from zero:** the saved prompt ends in the assistant primer the instruction replaces, so this prompt does not extend it |
| The prompt after compaction | Keeps the system prompt and tools, prefills the new user turn on | **Prefills from zero** |

This is why the summary prompt holds the tail when it can, and leaves it out only for room.

**So the ROADMAP's "reusing the system prompt's cached prefix" is false for Qwen.** Its recurrent state and convolution
windows exist only at the position saved (`Qwen.ixx`). The prompt after compaction is short, so this costs that prompt's
length and no more. It is corrected in `ROADMAP.md`.

Read from the code, and checked by the logged skips on Gemma and Qwen (`Notebooks/CompactionSummaries.md`).

### 5.2 Reasoning clears the prefix too

Clearing reasoning changes the rendering from the first turn that has any. Gemma's and Llama's templates render no past
reasoning, so clearing changes nothing and costs nothing. On Qwen, whose publisher default keeps it, it means a prefill
from zero of everything that remains. If that then summarizes anyway, the sequence pays a second prefill from zero of
the same text. Hence step 2: reasoning is cleared alone only when that is enough, and otherwise the summary reads the
prompt the state already holds, unless it lacks the room.

Unexplained: clearing alone took a real Gemma prompt from 12,347 tokens to 12,180, so what its template renders of past
reasoning is not quite nothing.

### 5.3 Tools on the summary prompt

Each template declares its tools at the head of the prompt: in the system content on Qwen and Gemma (`formatPrompt`),
and on Llama in the system header in the form Chat and Mila Desktop use, or in the first user turn by its default.
**A summary prompt with no tools therefore differs from the state almost at once: a prefill of the whole conversation,
at its deepest,** and then the prompt after compaction prefills the system prompt and tools again.

Measured (`Notebooks/CompactionSummaries.md`), reading the summary prompt from the state a compaction finds, at about 33K:
Gemma 4 12B Q4_0 3.2 s with the tools and 14.5 s without; Qwen 3.8 27B FP4 9.3 s and 34.3 s. With the tools kept, the cost is
the latest reply, its results and the instruction.

Keeping them has its own risk: the model may call one. A call is not run, and the instruction says not to. None of 26
summaries offered tools called one, on two models with a repository session's tools and no system prompt of Mila Desktop's;
8 of 32 did on Qwen under Mila Desktop's own prompt and tools, replaying a real conversation. With no tools offered, 0 of 7 did
(`Notebooks/CompactionSummaries.md`). So a call costs one short attempt and then the re-read this section prices, which is why
the tools stay for the first attempt and go for the second.

## 6. What the program sees

`AgentEvent` gains three kinds:

- **`CompactionBegan`:** a long wait is starting, with nothing streaming. The tokens the prompt holds.
- **`Compacted`:** `replaced_turns`, a span valid for the event alone, so the application can keep the turns the session
  no longer does (decision 6); the summary's text, empty if only reasoning was cleared; the turns dropped unread; the
  tokens before and after.
- **`CompactionFailed`:** why, in words.

`Response` gains `compactions`. `compact()` returns a `CompactionOutcome`: `Compacted`, `ReasoningCleared`,
`NothingToReplace`, `Failed` or `Stopped`.

An application times a compaction from `CompactionBegan` to `Compacted`, as it times a round; the measurements stay in the
consumer. The trace (`AgentTasks.md` 6.1) keeps these events later.

## 7. The apps

**Mila Desktop** stores `messages`, what the page shows, apart from `turns`, what the model reads
(`Desktop.SessionProcess.ixx` `saveConversation`). So the person's view survives untouched, a reopened conversation
continues compacted, and the window needs only a line for the two events. It keeps no replaced turns in this change:
retrieving them is `Future.md` *Compacted turns stay retrievable*.

**Chat** prints one `info` line for each, the same way it says it reached the limit, and `/compact` runs one when the person
asks. `/context` gains nothing. Its overflow message stays for what still reaches it.

**Neither carries a depth yet:** both leave the trigger absent until profiles exist (decision 3), and set `summary_tokens` to
6,000 or a quarter of the context, whichever is less, with thinking off where the template has the switch (decisions 7, 8). The
quarter leaves a short context room to answer.

## 8. Measurement and tests

**Summary lengths and prefill were measured first:** `Notebooks/CompactionSummaries.md`. Qwen at its default `xhigh` ran out of
an 8,192 token budget in 4 of 6 and took about six minutes; thinking off finished 4 of 4, at 2,180 to 4,581 tokens, in 2 to 4
minutes. Gemma with thinking off, its default, stayed under 1,000 tokens.

**Library tests, model-free where they can be:** the protocol renders the replaced form for each family's template,
Llama's single result included; a trigger not reached leaves the prompt byte-equal to no policy at all.

**On Gemma 4 12B Q4_0, with a program-set trigger** (decision 7):

- Compacts at each of the three boundaries; `kept`, the system prompt and the message reach the next prompt word for
  word; the calls in the tail keep their results.
- The model goes on: a session that passes the trigger finishes its work, and states a fact from before the compaction
  that the summary was told to keep.
- Each failure leaves the conversation byte-equal: a cut summary, a call in the summary, a stop, a summarizer that throws.
- A tail that holds the context does not compact again and again.
- `replaced_turns` is what the session held. Tools allowed for the conversation stay allowed.
- A later message's prefill is what section 5.1 says, read from the log as `AgentCuda` reads it.
- Qwen and Llama in the final run, as `AgentCuda` has them.

Each behaviour above gets a mutant, as the agent's tests have had.

**Real runs** (`Notebooks/CompactionSummaries.md`): Mila Desktop's session over real source, at a 16K context, with a rule and
a fact stated once in the first message. Gemma carried both across six compactions, once the instruction was changed for a
loss the earlier wording caused. Qwen carried both across three on the earlier wording, and on the next dead-ended after six;
that replay is what decisions 9 to 12 come from. As built, Qwen reached the end of the same session: seven compactions, the rule and
the fact kept, one summary failing at the last message with nothing refused. Gemma kept both across four, and lost the first function
asked about. Llama 3.1 8B never reached the trigger, so it has the tests and no real run past a compaction. The last resort has its
tests and no real run.

**The ROADMAP criterion is not met by this.** It is measured by the instruction-retention arm past the reliable depth,
which `Tools/ContextProfile` lacks. Order, decision 7: this mechanism; then the retention and tool-call arms (the
BACKLOG's prompt mismatch in that tool first); then profiles of the published models and the depths recorded.

## 9. Decisions

Decided 2026-11-03 (Todd, "Agreed", each as recommended). The first two change what was agreed earlier that day, once
the code was read.

1. **The tools stay in the summary prompt, none is run, and a call fails the attempt.** Reverses "summarize with no
   tools": without them each compaction re-reads the whole conversation (5.3). Subject to the measurement in section 8,
   which finds how often a model calls one.
2. **Reasoning is cleared alone only when that is enough, or when the summary lacks room.** Narrows "clear reasoning
   first", to spare Qwen a second prefill from zero (5.2).
3. **The oldest turns are dropped unread when a summary has no room, even with the tail left out.** Failing would end
   the session when it most needs to go on, and the turns still reach the application.
4. **An absent trigger means the context limit less `summary_tokens`, and that minimum always applies.**
5. **A summary counts against the response's time, and not its reply tokens.**
6. **A truncated summary is a failure, and a failure ends the automatic attempts for that response.**

7. **The policy may set template variables for the summary alone, and Chat and Mila Desktop turn thinking off with it, where
   the template has the switch** (Todd, 2026-11-03, after the measurement). On Qwen it costs a read from zero (the switch is at the prompt's head), 25 s more
   at 32K, against minutes saved and summaries that finish. What it does to a summary's quality is unmeasured, here and
   elsewhere: the mechanism's test on carrying on from a summary answers that. The library's default stays the session's own.
8. **Chat and Mila Desktop start with `summary_tokens` at 6,000** (Todd, 2026-11-03), above the longest thinking-off summary seen (4,581), from a small
   sample, and revisited with the mechanism's runs. It is an application's number, not the library's, so it hardens nothing.

Decided 2026-11-04 from a replay of the Qwen session that dead-ended, on Todd's direction to measure each proposed fix or
give a best guess (2026-11-03, `Notebooks/CompactionSummaries.md`):

9. **A summary that calls a tool is written again with no tools.** Narrows decision 1, whose measurement it answers: 8 of 32
   called one on the replay, 0 of 7 with none offered, every one of those under 4,096 with the rule and date kept.
10. **The library's instruction names a length, two fifths of `summary_tokens` in words.** With a length named, 22 of 23 fit
    4,096; with none, 5 of 8. Two fifths is arithmetic: Qwen wrote about two tokens to a word, which leaves a model a fifth to
    run over. It does run over: asked for 1,638 words it wrote up to 2,063, and one of those missed the budget. So this makes a
    cut rare and does not prevent one.
11. **A summary the earlier one already holds whole is written again, and a second is a failure.** 4 of 31 were that: everything
    since the earlier summary gone, and the rule and date still there, so no other check sees it. This catches the whole copy
    only, and not the partial one, which is every other summary measured and carries forward what it copies.
12. **The last resort of section 4.** Extends decisions 3 and 6. **A best guess, not a measurement:** nothing replayed reaches
    it, since decisions 9 and 10 stop most of the failure it is for. Its reason is the dead end measured, a session refused for
    good.

Not chosen: leaving the unanswered message out of a summary prompt whose tail has no room. It was the first theory of the tool
call, and calls were more common with it out, 3 of 5 to 1 of 5. Not chosen: a larger budget than a quarter of a short context.
With a length named, nothing needed it. Not chosen: asking the model to rewrite an earlier summary rather than copy it, which did
not stop it copying.

Still open, and not for this change: the KV cache format in the depth field (decision 3 of 2026-11-03).

## 10. Costs and risks

- **A summary loses things.** It keeps a conclusion and loses the evidence; that is `Future.md`'s entry. The retention arm
  measures instructions, not everything a session knew.
- **Library-written text.** The default instruction and the heading are in front of the model, and change what it does.
  They are measured, and the program may replace the instruction. The sentence that says turns are gone is not measured.
- **The last resort loses work without a summary.** The application is told and has the turns; the model has one sentence.
- **`kept` costs on every compaction,** and a long list is the program's own to answer for.
- **A second summary summarizes a summary,** and what is lost compounds. It did on Gemma: a rule from the first message was
  gone at the fourth compaction, because the model obeyed it instead of recording it. The default instruction now asks for
  every standing instruction, an earlier summary's included, quoted as an instruction and not applied to the summary, and
  the same run then held it through six. That is one rule and one fact in one session; the retention arm measures it properly.

## 11. Relationship to other specs

- **`AgentTasks.md`** 7 is the seam this keeps: `kept`, and the between-round boundary.
- **`ContextProfile.md`** gives the depth that later sets the trigger.
- **`Direction.md`** 4.2 rule 1 names `compact`, and 4.3's session holds the policy.
- **`ToolCalling.md`** is unchanged: no call in a summary passes the gate, because none runs.

Modules, one type each, under `Lib/Src/Agents/`: `Agents.CompactionPolicy`, `Agents.CompactionOutcome`.
