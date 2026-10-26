# Tool Calling

How a model's tool calls run through `Mila::AI`: what the Mila library provides and what an application brings.

**Status:** design, agreed 2026-10-25 (Todd), conceptually and with no release attached. Built to it at `+97`:
`Mila::AI::Agent` (`Mila/Src/AI/Agent.ixx`) with tools registered as a callable and a schema, the ask hook and its
three answers, no hook meaning deny, a call of a tool not offered or a tool that throws answered with an error result,
and a stopped reply running no waiting call. Built at `+98`: section 4's limits per response (`AgentLimits` --
rounds, time, reply tokens, none set by default), a stop or the time running out asking a running call to stop, and
a result the context cannot hold cut to what fits and the model told so; Chat and the desktop app's session run
their calls through the agent and hold no loop of their own. Built at `+99`: section 4's splice -- within a response
each round continues from the reply as generated and the results as the template frames them
(`Protocol::continueWithToolResults`), and Qwen continues from the state its last generation left. The rest of
section 3 is not built. It replaces `ToolBroker.md` as the design of record for where tool calling lives; the code built to that
spec -- the broker in `Mila/Apps/Tools/Broker`, the client in `Mila/AI/`, the session's own handler map in Chat --
is refit to this one. Open decisions are in section 9.

---

## 1. The rule

**The Mila library provides everything a developer needs to build an application like the Mila desktop app**
(`ChatDesktop.md`). An agent's tools are the core of that app, so tool calling is the library's. Leaving it to
each application would either give the Mila app a capability the library lacks (`Direction.md` 3.3) or require a
second harness library, which would be Mila under another name.

Two further reasons put it in the library and not in a harness across a wire. A tool result's tokens can be
appended to the live KV cache, with no re-render and no re-tokenize (section 4), only when the loop runs in the
process that holds the model. And how each model writes a call and expects a result is the model's own grammar.

There are two pieces: **the Mila library**, which the application links, with `Mila::AI` a namespace and API inside
it on top of the model; and, at most later, a process outside it (section 7).

**Through `Mila::AI`, the application never runs a model's tool call itself.** It supplies tools and a way to ask
the person; Mila parses the call, checks it, runs it and puts the result back into the turn. An application that
wants to run calls its own way works at the model level (`Mila::Dnn` and the handle) and does so there: an explicit
choice to leave the structure, not a way around it.

## 2. Two levels

- **The model handle: the model's tool grammar.** Writing tool definitions into the prompt in the form the model
  was trained on; parsing calls out of the stream as the tokens arrive -- where a call starts and ends, its name and
  arguments, several calls in one turn, a call that is malformed or cut off; writing a result back in the model's
  own form; separating the stream into thinking, answer text and calls in progress, so an application can show text
  live and know when a call is forming. From outside it is family-blind: the caller sees a call, a name and
  arguments, never a family. Whether a model calls tools, or several at once, is a declared capability
  (`ModelHandle.md`). The inference server works at this level: it parses a call and returns it, since the outside
  harness owns its tools.
- **The agent, in `Mila::AI`:** the loop of section 4 and the four pieces of section 3. Every call it parses runs
  through them.

## 3. The four pieces

### 3.1 Tool

- A name, a description, an input schema and, optionally, an output schema.
- **Declared needs:** whether it reads files, writes files or uses the network, and which areas it touches. They
  are facts the permissions of 3.3 decide on, not hints to the model.
- **Names carry their source** (`<source>.<tool>`), so two sources cannot collide.
- **The description is prompt text the model reads.** It is pinned when approved; a changed description is asked
  about again.
- **Registration:** a C++ callable and its schema, registered on the agent.

### 3.2 Execution

How a call reaches the tool. **Initially in-process only:** a function of the application, called in its process.

- **The arguments are checked against the schema before anything runs.** The model's output is untrusted input.
- A timeout, cancellation, and a cap on the result's size.
- **A failure reaches the model as an error result,** never as an exception into the application, so the model can
  recover.
- Results are text or structured data, and later images for models that see.

### 3.3 Permissions

A gate that every call passes; inside `Mila::AI` there is no path to execution around it.

- **In:** the agent, the tool, the arguments and the tool's declared needs. **Out:** allow, ask or deny, and with
  allow a **grant** -- which folders, read-only or read-write, whether the network, the time and size caps.
- **Rules match on arguments,** not only on the tool: a read inside the project folder can be allowed while one
  outside it is denied.
- **Ask goes through a hook the application supplies.** Mila decides that the person must be asked and what they
  are shown: the tool's name and summary, every argument in full, and what it will touch. **The question is built
  from Mila's own data, never from text the model wrote.**
- **Answers:** once, for this conversation, or deny. Making a rule permanent is a change to the rules, never an
  answer to a question.
- **No hook means no one to ask,** so ask is deny.
- **A denial reaches the model as a result** saying the person declined, so it can continue. A stopped reply
  cancels a pending question.
- **A record of every call** -- the decision, who decided, duration, sizes -- reaches the application as an event.
  Where it is kept is the application's.

### 3.4 Containment

What holds a call to its grant. Permissions decide; containment enforces. A grant is worth nothing unless the thing
running the tool holds it there.

**For now, containment is the grant object.** Mila passes the tool its grant as an object it acts through: the tool
opens files and makes connections through the grant, and the grant checks each at that moment. A check made ahead
of the tool, on the text of its arguments, is not enough, since what a name leads to is known only once it is
opened.

- **Files.** The grant opens the path, then compares the opened file's final path -- after every link, junction and
  mount point is followed -- with the granted folder's own final path, and closes and refuses anything outside it.
  Before opening it refuses names that are not plain paths: alternate data streams (`name:stream`), device names
  (`CON`, `NUL`, `COM1`), the `\\?\` and `\\.\` forms, and network (UNC) paths. Short 8.3 names and differences of
  case resolve to the same final path and are caught there. A new file is checked through its folder, which must
  itself resolve inside the grant.
- **Network.** The grant resolves each host name itself and refuses private (RFC 1918 and IPv6 unique-local),
  loopback, link-local and unspecified addresses. It checks the address it actually connects to, not a name, so a
  name that resolves differently a second time gains nothing. A redirect is a new connection and is checked again.
- **What it holds.** A tool that works through its grant is held to it. One that opens files directly is not; an
  in-process tool is the application's own code, and that is the application's choice.
- **Mila never claims containment it does not enforce.** A tool that would need more -- a shell, another party's
  server, code an agent wrote -- is not offered until an implementation exists that enforces its whole grant.

The checks are the hard, security-critical part, which is why they are Mila's and not each application's: written
once, and each tested first reaching its target with the check removed, then refused with it.

## 4. The loop and the splice

- Generate; when a call completes, pause, run it through section 3, append the result's tokens to the live cache,
  and continue the same turn, until the model gives its answer.
- **Limits per turn:** the most tool rounds, the total time, the reply budget. Stopping a reply cancels a call that
  is waiting or running. *Decided 2026-10-25 (Todd):* the library sets none by default -- the context bounds a
  response that has none -- and an application sets its own; Chat keeps the four rounds it has always run, a number
  no measurement set. The reply budget counts every round's tokens. The time the person takes to answer a question
  is not the response's. The calls of the last round allowed are answered as not run, since no round would read
  their results. A running call is asked to stop through a `std::stop_token`; an in-process tool can only
  cooperate, and what it returns after the stop is not given to the model.
- **A result larger than the context left** is cut or refused, and the model told which; it never silently
  overflows. *Decided 2026-10-25 (Todd):* cut, by tokens, to the first that fit, with a note of how many of how
  many; refused only when none fit. The cut keeps the prompt inside the context and reserves nothing for the
  answer, so the result cap of 3.2 is what keeps an ordinary result small.
- **The splice.** *Decided 2026-10-25 (Todd):* within a response the model continues from its own tokens, not from
  a rendering of them. A round's prompt is the last round's prompt and its reply as generated, followed by the
  results as the template frames them after a call -- the model's protocol gives that text, tested against the
  template's full rendering -- so a round prefills only the results it adds. A reply whose last call is left open is
  rendered instead, so the model never reads a call it did not finish followed by its result. Across responses the
  conversation is rendered: Gemma and Llama prefill from where the rendering first differs from what they hold,
  such as the reasoning a template drops, and Qwen from the end of its last reply when the rendering extends it,
  else from the end of the last round's prompt. Measured in `Direction.md` 5.3. The model ends a reply on the token the template opens its results with (Gemma's
  `<|tool_response>`, Qwen's `<|im_end|>`, Llama's `<|eot_id|>`), and decode-ahead has already put that token in the
  cache, so the results' first token is never prefilled.
- **Prefix reuse is the model's property**, read from the handle: a rewind to any position, or for a model with
  recurrent state a return to a position it saved (`Direction.md` 5.3); Qwen also continues from the state its last
  generation left when a prompt extends every token that state holds. The loop does not know which family it drives.
- **The conversation is the agent's;** its history and the cache always agree.

## 5. What the application brings

- **The tools.** Mila ships none (Todd, 2026-10-25). The Mila desktop app's tools for files and the web are its own
  code, the first written through grants, and a developer's program could write the same.
- **Its default rules,** and where the person's rules are kept and how they are edited: a file, a settings page, a
  command.
- **How a question looks:** a window, a console line, or nothing for a headless program.
- **The project folder,** chosen with the person, which becomes the files grant.
- **Where conversations are saved.** Mila gives a conversation as data it can restore.

## 6. Later

- **MCP, client and server** -- not initially. A client would add local servers started as processes over stdio
  and remote servers over HTTP, whose tools join the same registry, so the loop and the gate treat them like any
  other; everything a server says is untrusted, and a changed tool list withdraws its approvals. A server would
  let an application offer its tools to other agents.
- **Tools in a process of their own** arrive with the client: such a tool is an MCP server Mila starts over stdio.
  The approval round trip MCP specifies for one -- its input-required result -- is `ToolBroker.md` section 5.
- **A sandbox** under the same containment interface, enforcing a grant whole. Microsoft's execution containers are
  kept out of Mila for now (Todd, 2026-10-25).
- **Remote servers' sign-in tokens,** held where neither the model nor the agent's text can reach them.

## 7. The broker

A per-user broker is not a fifth piece. It is sections 3.2 to 3.4 hosted in a process of their own, which makes it
an application built on the library, and it is needed only when the rules must hold against the application
itself -- one set of rules over agents someone else wrote, under an account those agents cannot control. Rules
inside the application protect the person from the model, steered by a hostile page or file; they cannot protect
the person from the application, which can skip its own. `ToolBroker.md` records the broker as designed on
2026-10-24.

## 8. Layering

The tool pieces need nothing of the model, and the model needs nothing of them; only the loop joins them. A program
that wants tools alone pulls in no CUDA. What the pieces add is JSON for schemas and arguments and the operating
system's file and socket calls for the grant; HTTP, TLS and process launching come only with section 6.

## 9. Open decisions

1. **Several calls in one turn:** run together or in order. Reads and writes may need different answers.
2. **Rules after untrusted content.** Once a fetched page or a file the person did not write enters the
   conversation, whether rules tighten -- for example, ask before any write that follows.
3. **Connections through the grant.** How an application's HTTP client makes its connections through the grant,
   so every connection, a redirect's included, is checked.
4. **Where `Mila::AI` sits in the build.** *Directory decided 2026-10-25 (Todd): `Mila/Src/AI/`, built into the
   Mila target* (`ModelHandle.md` 3.5). Still open: whether the tool pieces become a target of their own, so a
   program wanting tools alone needs no GPU stack -- asked when such a program exists.
5. **The record of a call:** the event's fields, and whether arguments are kept whole.

## 10. Relationship to other specs

- `Direction.md` 3 and 8 -- the layering and decision 6, which point here.
- `ModelHandle.md` -- the handle the grammar of section 2 lives on, and its declared capabilities.
- `ChatDesktop.md` 4.5 -- the desktop app's question and answers, which this design's hook presents.
- `ToolBroker.md` -- the earlier design; its approval round trip informs section 6, its checks are section 3.4's.
- `Notebooks/LocalToolService.md` -- the reading behind both: placement, identity, secrets, the round trip
  measured, and prior art.
