# A Local Tool Service

**Status:** Evidence, 2026-10-23 to 2026-10-24. Its decisions (sections 8 and 8.5) are carried to the design,
`ToolBroker.md`, and to `Direction.md` section 8, decision 6. It records the proposal, what was read to test it,
and the first measurement.

## 1. The question

Every tool call a Mila model makes is executed by whichever program is hosting it. Chat runs its own
(`registerTool` in `Mila/Apps/Chat/Src/Chat.ixx`, one stub tool registered in `main.cpp`); MIS executes none and
returns each call to its client (`MilaISCodexAgent.md`); a developer's program would write its own. Todd's
proposal (2026-10-23): one common, secure local service that executes the tool calls of Mila's own apps and of
end users' apps alike.

Two questions follow. Where does it live: in the library, in a DLL, or in its own process? And does putting it
anywhere but in-process cost the dispatch speed that the compiled-in registry was chosen for?

## 2. The shape proposed

**A broker process, and a thin client in `Mila::AI`.** Not in `Mila/Src`, which "knows nothing about
conversations, tools, agents or applications" (`Direction.md`). Not a DLL: code in a DLL runs with its host's
identity, so it cannot be restricted below it, and a fault in a tool lands in the process that holds the model,
the device and the conversation.

- **Front: MCP.** The broker is an MCP server that apps reach through `Mila::AI`'s client; any other MCP client can
  use it too, which is what makes it a service for applications that are not Mila's. MCP is stateless since its
  2026-07-28 revision (every request carries its own `_meta`; no protocol session), which suits a broker.
- **Back: restricted workers.** Each third-party MCP server runs in a sandbox of its own. Mila's own native tools
  (file read, grep, patch apply, the shell) are workers Mila writes, under the same restrictions. The broker
  runs no tool code itself.
- **One place for policy:** which application may call which tool, the tool schemas as approved, timeouts,
  result-size caps, and the audit trace `MilaProductFamily.md` asks of the agent.
- **Lifecycle:** one per user, started on demand, reached over a named pipe (a Unix socket elsewhere) that only
  that user can open. Not a Windows service, which runs as another account.
- **The compiled-in registry stays**, as the in-process path for tools that need no isolation. The result of a
  broker call returns to the client, which splices it into the live cache as today (`Direction.md`, *In process*).

## 3. What it would change

- **`Direction.md` section 8, decision 6:** "Tool registration: a compiled-in registry ... Declarative subprocess
  tools alongside it remain open for section 6". This proposal enters through that open item; it does not
  reverse the decision.
- **`MilaProductFamily.md`, *What a tool is*:** it leans toward the compiled-in registry because subprocess tools
  "forfeit" microsecond dispatch. Section 5 tests whether that holds for a resident process, as opposed to one
  started per call.
- **Applications:** an app's tool loop calls the client instead of its own handler map. Chat's approval gate stays
  in Chat (section 6.1).

## 4. What was read

External behaviour, read 2026-10-23:

- **MCP tools** (`modelcontextprotocol.io/specification/latest/server/tools`, revision 2026-07-28): "there
  SHOULD always be a human in the loop with the ability to deny tool invocations"; clients "MUST consider tool
  annotations to be untrusted unless they come from trusted servers"; clients should validate results, time out
  calls and log usage. Tool names are unique only within one server, so an aggregator prefixes them.
- **MCP security best practices** (same site, *Local MCP Server Compromise* and *stdio Transport Security in Proxy
  Scenarios*): a client should "execute MCP server commands in a sandboxed environment with minimal default
  privileges"; a proxy that spawns servers over stdio is an escalation path if a client is compromised, and
  should sandbox what it spawns and run with least privilege itself. The broker is exactly that proxy.
- **AppContainer isolation** (Microsoft Learn, `secauthz/appcontainer-isolation`): file and registry access only
  where granted, network access granted per kind (internet, intranet, server), kernel objects and windows
  isolated from other processes, and an identity unique to each user and application pair.
- **Job objects** (Microsoft Learn, `procthread/job-objects`): memory, CPU and time limits over a process tree;
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` ends every process in the job when the last handle closes; children join
  the job unless a breakaway limit is set. **Two caveats from the same page:** children created through
  `Win32_Process.Create` (WMI) do not join the job, and security limits must be set per process, not per job.
- **Prior art:** gateways that aggregate MCP servers behind one endpoint exist (for example `mcp-gway`, which binds
  loopback and screens local commands through an allow-list), as do Docker- and microVM-based sandboxes for
  tool code. The consistent point across them is that a gateway's routing and allow-lists do not replace
  isolation of the tool process itself. Read from project pages and blogs only; none was run.

Not read yet: Linux (bubblewrap, Landlock, seccomp) and macOS sandboxing.

## 5. Phase 1: the cost of a round trip

**Gate, written before the run** (`D:\Claude\tool-broker-probe\gate.md`): a warm round trip to a resident worker
over a message-mode named pipe — 256-byte request, 1 KiB reply, 20,000 calls after 1,000 warm-up — has a median
under 100 us; starting a process per call (created suspended, assigned to a kill-on-close job object, a 1 KiB
reply read from its stdout, waited for) has a median over 10 ms. Forced failure: arm B must exceed arm A on every
run.

**Result**, 2026-10-23, Todd's development machine, four runs from one build (`cl /O2`), nothing else running:

| Arm | Median (us) | p99 (us) | Max (us) |
|---|---|---|---|
| A. Warm pipe round trip | 6.8 to 7.2 | 9.7 to 20.0 | 151 to 201 |
| B. Process start per call | 13,786 to 13,862 | 14,915 to 17,235 | 15,348 to 20,565 |

The gate passes on every run, and B exceeds A by three orders of magnitude.

**Reading.** What subprocess tools forfeit is process start, not crossing a process boundary. A broker on the
path is two hops (application to broker, broker to worker); by arithmetic, about 14 us at the median, before
any MCP framing or JSON parsing. That is small against one decode step, which takes milliseconds.

**Not covered:** the worker was not in an AppContainer; there was no JSON or MCP framing; the machine was idle;
Defender's effect on arm B was not separated out. Phase 2 repeats arm A with each of those added.

## 6. Open decisions

### 6.1 Who approves a call

- **The calling application shows the prompt.** MCP already has the mechanism: the broker answers a call that
  needs approval with an input-required result, the application asks the person, and the call is retried with
  the answer. The approval is only as trustworthy as that application.
- **The broker shows its own prompt.** The approval cannot be forged by the application, but the broker needs a
  window, and a headless agent has nobody to show it to.

**What decides it is the threat model.** The broker defends the machine against two things: a model being steered
by what it reads (prompt injection), and a tool server that is malicious or wrong. It does not defend against
another program running as the same user: that program can already do anything a tool could, without the
broker. Under that model, an application showing the prompt is sound, and the second option buys protection
against an attacker who has no need of the broker. *Leaning:* the broker holds the policy (allow, ask or deny,
per application and per tool), and the application presents "ask". A headless caller has no "ask"; anything not
"allow" is denied.

### 6.2 Telling applications apart

Any process running as the user can open the pipe. The broker can read the caller's process id
(`GetNamedPipeClientProcessId`) and from it the program's path. Under the threat model in 6.1 this is attribution,
not authentication, and attribution is what the policy needs: "Chat may write files under this folder; this
script may only read." *Leaning:* identify by program path, with each application registered once by the user
before it gets anything beyond the defaults. A per-application token is the alternative if attribution by path
proves too loose.

### 6.3 Giving a sandboxed tool the user's files

An AppContainer worker reads only what has been granted to its identity. Granting a project folder means adding
an entry to that folder's access-control list, which changes the user's files' permissions, and has to be undone.
Less-privileged AppContainers restrict further. How a file tool or the shell is given a folder, and taken off it,
decides whether the Windows sandbox is usable for coding work at all. Not tested.

### 6.4 Where it lives and what it is called

Under `Mila/Apps/`, as a shipped application with a `mila-` command (`Notebooks/MilaCommandLineReview.md`). Name
not chosen.

### 6.5 Scope

The client is part of `Mila::AI` (`Direction.md`), so this lands with or after v0.22.0, not in v0.21.0. It sits
on two open items in `Mila/Issues/Vnext.md`: tool calling has never been driven end to end on Llama, and any
response containing a bracket enters Chat's tool-call parser.

## 7. Next phases

Each phase's gate is written before its run.

- **Phase 2:** arm A with the worker in an AppContainer, MCP framing and JSON parsing on both hops, and the broker
  in the middle. Gate: median under 100 us end to end.
- **Phase 3:** the sandbox's escape paths on Windows. The WMI route out of a job object, network with no
  capability granted, a file outside the granted folder, and a child process. Each must fail; each test is shown
  to pass against an unsandboxed worker first.
- **Phase 4:** the same on Linux, after reading the documentation in section 4's *Not read yet*.

## 8. 2026-10-23, later: the broker knows the user and the agent

**Requirement (Todd):** the broker is the holder of the security rules, and it must know who the user is and who
the agent is. Every rule and every line of the audit trace is keyed on that pair. This replaces the leanings in
6.1 and 6.2, which keyed rules on the application alone.

### 8.1 The user

On one machine the operating system already authenticates the user: the broker reads the token of the process at
the other end of the pipe (`ImpersonateNamedPipeClient`, or the client's process id to its token) and gets the
user's SID. Nothing to build, and nothing an application can forge.

That holds only while the user is the account the caller runs as. **When a server runs the agent on behalf of a
person who is not that account** -- MIS serving a remote client -- the server is the caller and the user is
someone it vouches for. The broker would then trust the server to state the user, and the rules for that are
delegation: the server holds a credential of its own, and the user's identity arrives as a token issued for the
broker, never one passed through from elsewhere (MCP's authorization guidance forbids token passthrough). Open
whether the broker serves that case at all (8.4).

### 8.2 The agent

An agent is more than a program: one Chat process can run two conversations with different models and different
permissions. *Proposed definition:* an agent is a configuration the user registers with the broker -- the
application, the model, and the policy it runs under -- and the broker issues it a credential. The application
presents the credential on every call; the broker looks up the agent and its rules.

How much that credential proves depends on who can read it:

| Agent's process | What the credential proves |
|---|---|
| Ordinary process, same user | Attribution. Another program running as the user can read the credential from the agent's memory or files and present it |
| Process isolated from the user's other programs (AppContainer, or the packaged-app equivalent) | Identity. Other programs cannot reach the credential |
| Process the broker itself started, credential handed over at start | Identity, and the broker knows the process is the one it started |

The third row is the strongest, and Chat's desktop window already runs its session as a separate process
(`ChatDesktop.md`), so the broker starting that session is not foreign to the design. Not decided.

**The model is asserted, not proven.** The broker cannot see which weights an application loaded; a rule that
depends on the model rests on the application's word.

### 8.3 Why agent identity must be real, not attribution

Section 6.1 argued that impersonation gains an attacker nothing, because a program running as the user can do
anything a tool can. That stops being true once the broker holds anything the user's programs do not: OAuth tokens
for remote MCP servers, API keys, or grants to folders the user's own account cannot reach. An agent credential
that can be lifted then hands those over. So either the broker holds no secrets, or the agent's credential has to
come from the second or third row of the table.

### 8.4 Open

1. Does the broker serve agents acting for someone other than the machine's user (MIS's remote clients), or only
   the local user?
2. Will the broker hold secrets (8.3)? That decides which row of the table is the minimum.
3. Which isolation an agent's process needs, and whether the broker starts agent processes.

### 8.5 Answers and recommendations, same day

1. **Local user only** (Todd: MIS serves the session harness and does not take part). MIS stays a wire; its
   clients' own harnesses run their tools. Serving people other than the machine's user would make the broker a
   network-facing, multi-tenant authorization service -- a different product.
2. **Secrets will come eventually** (Todd). Recommended rule from the start: a secret never reaches an agent or a
   model. The broker hands it only to the worker that uses it, so a compromised agent can make a call it is
   granted but cannot take the token away.
3. **Who starts the agent: the application** (recommended). On a normal Windows desktop, as understood here and
   to be tested in phase 3, a process can open any other unsandboxed process running as the same user, read its
   memory and run code in it. Starting the agent from the broker therefore does not stop another program running
   as the user from taking it over, and buys little over the application starting it. The broker identifies the
   agent instead by the process at the other end of the pipe -- its program and its signature -- with the agent
   within that process declared by the application.

**What secrets actually force is where the broker runs.** A broker running as the user protects against the model
and against tool servers, but not against other programs running as the user: they can read the broker's memory
and decrypt what it stores for the user. Protecting secrets from them needs the broker in an account of its own
(a service), which costs an administrator install and makes the user check of 8.1 a real boundary between
accounts. *Recommended:* a per-user broker that holds no secrets first; secrets arrive with the move to its own
account, and that move is their gate. Not decided.

**Decided 2026-10-24 (Todd):** the first broker runs per user and holds no secrets, and its move to an account of
its own is the gate for secrets; the application starts its agent, and the broker identifies the agent by the
signed program at the other end of the pipe. Carried to `ToolBroker.md`.

## 9. 2026-10-24: a harness that rewrites itself

Todd added *Harness^2: Recursive Agent Harnessing for an Open World* (Xu et al., arXiv 2610.14224, 2026-10-09).
Read: the abstract and the HTML body; the prompts in its Appendix H are encoded in the page and were only partly
legible.

**What it does.** A harness is eight files around a frozen model -- system prompt, guardrails, memory, skills,
tool guidance, scripts, subagents and a middleware plugin (its section 2.1). An editor agent proposes two edited
harnesses for a task, runs each in a fresh container, compares their behaviour with the current one and
composes the result (2.2.1, Appendix B); the edits that the comparisons support go to a library that later tasks
read (2.3). It does not discuss security, permissions or human review of an edit. Its editor is held only by
prompt rules: evidence is data, not instructions, and it must never run a script or plugin found in a candidate
harness.

**What it means for the broker.** Mila does not build this; it is the kind of agent the broker must stay safe
under:

1. **The rules cannot be part of anything an agent edits.** The paper's guardrails and its tool-loop plugin are
   files the editor rewrites. The broker's policy lives where no agent's process can write it, which is a
   further reason for the broker's own account (8.5).
2. **A script an agent writes is a new tool of unknown origin.** It runs as a broker worker, sandboxed, recorded
   as written by that agent, and asked before its first run -- never trusted because a library entry adopted it.
3. **A probe is a disposable sandbox.** Each of the paper's rollouts runs in a fresh container; locally, that is
   a worker created for one run and destroyed after it, so two candidate harnesses cannot see each other's
   effects.
4. **"Never run a script found in a candidate" is enforced, not asked for.** The paper gives it to the editor
   as a prompt rule; under the broker the editor's agent has no grant to run it.
