# Agent Tasks

How an agent carries a task to its end over many responses, knows how it ended, and leaves a record of every step:
the task, its budget, the autonomy policy, the trace and the folder that holds them.

**Status:** agreed 2026-11-03 (Todd, decisions in section 10). Not scheduled, no code, apart from compaction's seam
(section 7), which is part of v0.21.0's compaction item. This carries
`Direction.md` 6.1 and `MilaProductFamily.md` *Agentic* into today's `Mila::Agents`.

---

## 1. What is missing

Read from `Lib/Src/Agents/` on 2026-11-03:

- **An agent works only inside `respond()`.** It generates, runs calls and continues until a reply makes no call or a
  limit ends the response, on the caller's thread. Between responses nothing happens. Within one, nothing needs a
  person if the rules allow the calls.
- **The goal is the text of a user message.** Neither "task" nor "workspace" appears in `Lib/Src/Agents/`.
- **A reply with no call ends the response, whatever the reason.** Finished, gave up and asked a question all look the
  same. `Response` says why generation stopped, not whether the work is done.
- **A full context is a dead end.** A message that does not fit returns `ContextOverflow` and is dropped; a response
  that overflows partway leaves its turns in the conversation, so the next message fails too. Compaction, the v0.21.0
  item, is what removes this.
- **Nothing sees a loop.** A model repeating a failing call runs until a limit or the context ends it.
- **Limits are per response.** `AgentLimits` holds no budget across responses.
- **The record leaves with the event.** `AgentEvent` is the record of a call, and where it is kept is the
  application's (`ToolCalling.md` decision 5). Nothing is written down by the library.
- **Mila Desktop has already written part of this for itself.** `Desktop.TaskPlan` holds the steps of Get Things Done,
  and the loop never reads them. Chat and a developer's program cannot reach it, which is the gap `Direction.md` 3.3
  names. `MilaDesktop.md` leaves open what Get Things Done needs of `Mila::Agents`; this is the answer proposed.

One thing is already right. A new response re-renders the conversation and the model skips what its KV cache holds:
`reuse_kv_prefix` is on by default and the agent leaves it on, read in Gemma's and Qwen's `generate` (Llama's not read).
So the claim "without re-reading its own history" is true across responses today, until something changes the prefix.
Compaction changes it by design (section 7), and so does a template that drops past reasoning.

## 2. The concepts

An agent stays a model and a session (`Direction.md` 4.3). Two more objects stand beside them, and neither goes inside
the agent, which ends with its model.

| | Answers | Lives |
|---|---|---|
| Model | What thinks | Until released |
| Session | What the program gave it, and what was said | Across models |
| **Task** | What it is for, what it may spend, how far it has got | Across responses, compactions and models |
| **Task folder** | Where the record of the work lies | At least as long as the task |

A program that never makes a task pays nothing and sees no change: `respond()` stays as it is, and `run()` is built on
it. Chat answering a person is `respond()`.

## 3. The shape

```cpp
Mila::Agents::Task task( "Make the parser tests pass." );

task.withBudget( { .time = std::chrono::hours( 1 ), .reply_tokens = 200'000 } )
    .withCheck( []( std::stop_token stop ) { return runTheTests( stop ); } )
    .withFolder( "D:/work/parser-task" );

const Mila::Agents::TaskOutcome outcome = agent.run( task, on_event, stop );

switch ( outcome.end ) { /* section 4.2 */ }
```

- **`run` blocks on the caller's thread, as `respond` does.** A program that wants it in the background owns the
  thread. Mila has no scheduler and no timers: serving many and scheduling are non-goals (`Direction.md` 9).
- **The task is passed by reference and changes as it runs.** It holds what was spent and how far the plan has got,
  so another model continues it: release the session, create the next agent, `run` the same task. This is why the
  state is not the agent's.
- **`run` is one new method on the agent.** `Direction.md` 4.2 rule 1 lists the contract, so that list grows by one.

## 4. How a run goes

```
add the goal to the conversation, unless the task is being resumed
loop:
    response = respond( message )        // limits per response are unchanged
    record every event in the trace
    add what it spent to the task
    end the run if a budget is spent, stop was asked, or the policy says stuck
    finish_task called?       run the check: passed -> Finished; failed -> its words are the next message
    ask_person called?        the asker answers, or the run ends NeedsPerson
    reply with no call?       the next message asks it to go on or finish; counted as a step with no progress
```

### 4.1 How the model says it is finished

**A tool, `finish_task`, with a required `summary`, offered for the length of the run.** A tool works in every family's
template and is read by the parser the agent already has, where a phrase in a reply would be a guess. It passes the
gate like any call, allowed by a rule the run adds, since it changes nothing outside Mila.

**The model's word is only a claim.** With a check, the program's check decides, and a failure goes back to the model
as the next message. With no check, the run ends `ClaimedFinished`, never `Finished`, so a program can tell what it was
told from what was shown. The check belongs to the task and not to the policy, which answers the question
`MilaProductFamily.md` open decision 3 left: the policy tests for an end, and the program owns what counts as one.

**`ask_person`** is the escalation `Direction.md` 6.1 asks for, available under any policy: with an asker it is answered
and the run goes on; with none the run ends `NeedsPerson`, the question in the outcome, and the program runs the task
again with the answer.

Whether models use these two tools well is unmeasured, and it is the design's largest risk (section 9).

### 4.2 How a run ends

`TaskOutcome` holds the end, the model's summary or question, and what was spent.

| `TaskEnd` | Meaning |
|---|---|
| `Finished` | The model said so and the program's check passed |
| `ClaimedFinished` | The model said so, and the task has no check |
| `NeedsPerson` | It asked, and no one can answer |
| `Stuck` | The policy found no progress (section 5.2) |
| `BudgetSpent` | A budget of the task ran out; which one is named |
| `ContextFull` | The conversation cannot go on, and compaction could not make room |
| `Stopped` | The caller's stop |

This is the split `Direction.md` 6.1 measures: stopped when done, and stopped when stuck.

### 4.3 Stop and resume

A stop lands as it does in `respond()`: every call keeps a result, and the next response renders the conversation again,
so the KV cache never describes a turn that did not happen. That already meets `MilaProductFamily.md`'s "on a turn
boundary, never mid-splice", with nothing added.

A task is resumed from its folder (section 6). The conversation comes back through `Session::openConversation`, which
forgets the tools allowed for the conversation, so a resumed run asks again. A resume prefills: resuming with none is
`Direction.md` 6.4, *Memory*, and this design does not need it.

## 5. The policy

A runtime object, as `MilaProductFamily.md` decided: `shouldContinue` over the trace.

### 5.1 Budgets

`TaskBudget`: time, generated tokens, responses and tool calls, each across the whole task. As in `ResponseClock`, the
time a person takes to answer is not counted.

**A run with no budget at all is refused.** `AgentLimits` can leave every limit unset because "the context bounds it".
Compaction removes that bound, so nothing else would end a task that never finishes. Mila invents no default number
(`MilaProductFamily.md` leaves the defaults open, and this keeps them the program's): the program names at least one
of time or tokens. A limit per response goes on applying inside a task.

### 5.2 Loop detection

On by default, from the two mechanical signals `MilaProductFamily.md` settled:

- **Repeated call identity:** the same tool and normalized arguments, with no state change between them.
- **No observable change:** a run of steps in which no call's result differed from what it returned before.

**Their limit:** the library sees calls and results, not the world. A hash of a result is what it can compare, so a
result that embeds a clock looks like progress, and a tool that changes something while returning the same words looks
like none. `ToolNeeds::writes_files` says a tool may write, not that this call did. How many repeats mean stuck is a
measurement, not a number to pick now.

**Thought similarity** stays deferred, as `MilaProductFamily.md` has it.

## 6. The task folder

**Named for what it holds, and not "workspace".** That word is taken in this library: 214 uses in `Lib/Src`, for the
scratch buffers of `GqaWorkspace` and `Qwen.DeltaNetBlock.Workspace`, and `Notebooks/Workspaces.md`.

```
<task folder>/
    task.json             version, goal, budget, what was spent, the plan, the end
    trace.jsonl           one line per step, written as it happens
    conversation.json     the session's conversation, written between responses
```

- **The program names it, and the library never chooses one.** No folder means no record and no resume, and `run`
  works without it.
- **It is the library's, not the model's.** It is written with plain file I/O, and no tool reaches it.
- **A folder a call can write is refused as a task folder.** Otherwise a call could rewrite the record of what it did,
  and `.mila` in a project is exactly that: a contained program's home, which the program can write. The check compares
  final paths against the grant's folders read-write, as the grant does for a file, when the run starts and whenever
  the session's grant changes.
- **It is not where the work happens.** The work is in the grant's folders, and the grant stays the one limit on what a
  call can reach. Keeping two agents out of each other's way in one project (a branch, a copy) stays the program's. A
  scratch folder for the model is a folder the program grants.
- **It holds no search.** Retrieval stores are an application's concern (`Direction.md` 6.4). `Future.md` *Compacted turns
  stay retrievable* can build on this folder later.

### 6.1 The trace

`Direction.md` 6.1: autonomy without the audit is a runaway. The trace is `MilaProductFamily.md`'s thought, call, result
and decision, as a **versioned format**, the only thing a person may ever see of an unattended run.

- One line per step, flushed as it is written, so a crash leaves everything before it.
- A call: tool, arguments, decision and who made it, duration, the result as the model read it, and the bytes before any
  cut, all of which `AgentEvent` already carries.
- Also: reasoning, what each round spent, each compaction and what it replaced, each check and what it said, each
  policy decision.
- Never a token stream. `AgentEvent::Token` is for the program's sink.

Its format has the same promise as weights on a disk (`ModelDistribution.md` *Compatibility*): what a published minor
wrote, the next one reads.

## 7. Compaction: the one seam for v0.21.0

Compaction is committed to v0.21.0 (`BACKLOG.md`, *A conversation that fills its context stops*); tasks are not. It
keeps "instructions verbatim", which today means the system prompt. **The goal is in a user message, so a summary would
replace it.** Two requirements on compaction keep this design open without any of its code:

1. **What is kept verbatim is a list the program can add to,** not the system prompt alone. A task fills it with the
   goal and the plan as it stands. Compaction stays mechanism: what a summary keeps is already the application's policy
   (`ROADMAP.md`).
2. **It runs between rounds, not only between responses.** With no round cap one response can fill the context by
   itself, as Build & Code's do, so a boundary between responses comes too late. The in-flight continuation is dropped
   when it happens, and the next round renders the conversation again.

Compacting costs a prefill of everything past the system prompt's cached prefix. It should appear in the trace with what
it cost, since prefill per turn is what `Direction.md` 6.1 measures.

## 8. The plan moves to the library

`Desktop.TaskPlan` becomes `TaskPlan` in `Mila::Agents`, with its two tools provided as the file and web tools are:
`propose_plan`, asked, and `update_step`, allowed. A program chooses whether to offer them. The plan belongs to the task:
it is kept through compaction and written in `task.json`.

Its contract stays as Mila Desktop has it, with one addition: `finish_task` with steps still pending goes back to the
model saying which, as a failed check does.

Get Things Done then runs on `run`, and its own plan is deleted. Its person stays present through the asker. The trace
gives the window nothing new it must show.

## 9. Costs and risks

- **Autonomy amplifies model weakness** (`MilaProductFamily.md`, *Honest Risk*). How often a model ends a task too early
  or never ends it is unmeasured. `finish_task` may be called too soon, or never. A check limits the first, and the
  policy limits the second. Neither makes a weak model good.
- **Unattended commands are not solved here.** `run_command` is always asked, and no rule allows it
  (`ContainedPrograms.md` 5), so a Build & Code task with no person stops at its first command. The container is
  experimental, Windows only, with known gaps. Whether a person may allow commands for a whole task is its own decision,
  for that spec.
- **Library-written prompts.** The "go on or finish" message and the tools' descriptions are text the library puts in
  front of the model, and they change what it does. They are the policy's to replace, and the defaults are measured.
- **Format promise.** A versioned trace and `task.json` are a compatibility surface from the first published minor.
- **Which models.** Suitability is measured (`agentic`), so `run` does not refuse a model by name or by tag. Whether it
  It reports the model's tag (decision 6).

## 10. Decisions

All decided 2026-11-03 (Todd; 2 to 7 "Agreed", each as recommended):

1. **The library owns tasks,** since Chat, Mila Desktop and a developer's program would otherwise each write their own
   (`Direction.md` 3.3).
2. **The compaction seam of section 7 is in v0.21.0,** as requirements on the existing item, with no task code in
   v0.21.0.
3. **`agent.run( task )`, a method.** It is the one obvious way to reach a task, and `Direction.md` 4.2 rule 1's list
   grows by it. Not chosen: a free `Mila::Agents::run( agent, task )`.
4. **A run with no budget is refused** (5.1).
5. **The name is "task folder"** (6). "Workspace" would be one word for two concepts.
6. **`run` works on a model not measured as agentic,** and the outcome reports the model's tag.
7. **The release stays after v0.22.0,** where `ROADMAP.md` holds it, since v0.22.0's claim is deployment. `Vnext.md`
   carries it.

## 11. Measurement and phases

The gate is `Direction.md` 6.1's: the same model and tools driven in process and through MIS by a foreign harness, on
one task set, reporting pass rate, turns, prefill tokens per turn, wall-clock, and termination split into stopped when
done and stopped when stuck. Models at their defaults, with Gemma 4 12B Q4_0 as the working model, and every model
in a final run. No number is promised until a phase has measured it.

1. **The seam:** section 7, inside the compaction item.
2. **The task:** `Task`, `TaskBudget`, `TaskEnd`, `TaskOutcome`, `run`, `finish_task`, `ask_person`. The first measurement:
   do models finish, and do they ask, on the task set?
3. **The record:** the trace and the task folder, then resume, then the task surviving a change of model.
4. **The policy:** loop detection, measured against runs known to perseverate.
5. **The plan:** `TaskPlan` into the library, and Get Things Done onto `run`.

Modules, one type each, under `Lib/Src/Agents/Tasks/`: `Agents.Task`, `Agents.TaskBudget`, `Agents.TaskEnd`,
`Agents.TaskOutcome`, `Agents.TaskPolicy`, `Agents.TaskFolder`, `Agents.TaskStep`, `Agents.TaskTrace`, `Agents.TaskPlan`.
`Mila::Agents` gains `run`, so `Lib/Src` changes only with Todd's direction, phase by phase.

## 12. Relationship to other specs

- **`Direction.md`** 3.2 and 6.1 own the claim; this is the design they point to. 6.4 *Memory* stays separate.
- **`MilaProductFamily.md`** *Agentic* holds the reasoning this carries forward; open decision 3 is narrowed by 4.1.
- **`ToolCalling.md`** owns the gate and the grant, which this uses unchanged, and the call's record, which the trace keeps.
- **`ContainedPrograms.md`** 5 keeps a command always asked (section 9).
- **`ContextProfile.md`** gives the reliable depth compaction triggers at.
- **`MilaDesktop.md`** 3.1 and its open question on Get Things Done change at phase 5.
