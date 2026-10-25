# Tool Broker

One local service that executes the tool calls of every agent on a machine -- Mila's own applications and other
people's -- under one set of rules, each tool in a process of its own, limited by the service's checks and, once
section 6.3's sandbox arrives, by the operating system.

**Status: replaced as the design of record by `ToolCalling.md` (2026-10-25, Todd).** Tool calling is `Mila::AI`'s
structure in the Mila library, every tool comes from the application, and MCP comes later; a broker is at most an
application built on those pieces (`ToolCalling.md` section 7). This document records the broker as designed and
built on 2026-10-24, and the code below is refit to `ToolCalling.md`. Its approval round trip (section 5) and its
path and address checks (section 6.2) carry into that spec; its placement, its client in Mila.AI and its first
tools do not.

**Was:** draft, 2026-10-24. The process, pipe and message layer is built and phase 3 is measured (sections 8,
9); so are MCP's tool methods, rules held in memory, the approvals of section 5, and Mila.AI's client. The checks
of section 6.2, where the rules are kept, the audit trace and the executable are not. The evidence behind it, and the reasoning for each decision, is
`Notebooks/LocalToolService.md`. **Ships in v0.21.0 on Windows** (Todd, 2026-10-24), behind the desktop app's tool
calling (`ChatDesktop.md`): the ROADMAP theme *The Desktop Agent*. Open decisions are in section 10.

---

## 1. What this is

Today each program that hosts a Mila model executes its model's tool calls itself: Chat through its own handler
map (`registerTool`, `Mila/Apps/Chat/Src/Chat.ixx`), a developer's program through whatever it writes. Nothing
holds the rules in one place, nothing restricts what a tool can reach, and no Chat asks before a tool runs.

The broker is a separate process that holds the rules and runs the tools. An agent asks it to call a tool; the
broker decides whether that agent, acting for that user, may; and if so, a worker process runs the call within
the limits of section 6 and the result comes back.

**Not in scope:**

- **The inference server.** MIS serves the client's own harness, which runs its tools; it takes no part in this
  (Todd, 2026-10-24).
- **People other than the machine's user.** The broker serves agents acting for the account they run as. An
  agent acting for a remote person would make it a network-facing, multi-tenant authorization service.
- **The library.** `Mila/Src` knows nothing of tools (`Direction.md` 3.1).

## 2. Principals

Every rule and every line of the audit trace is keyed on the pair **(user, agent)** (Todd, 2026-10-23).

- **The user** is the account of the process at the other end of the pipe, read from its token by the broker.
  The operating system authenticates it; no application states it.
- **The agent** is a configuration the user registers with the broker once: the application, the model and the
  policy it runs under. The broker identifies the agent's process by the **signed program** at the other end of
  the pipe (Todd, 2026-10-24); the application declares which of its registered agents a call belongs to, since
  one process may run several.
- **The application starts its agent** (Todd, 2026-10-24), as Chat's window starts its session process
  (`ChatDesktop.md`). On a normal desktop a program running as the user can take over another unsandboxed one,
  so the broker starting it would add little.
- **The model is asserted, not proven.** The broker cannot see which weights an application loaded. A rule that
  depends on the model rests on the application's word, and says so where it is shown to the user.

## 3. Architecture

```
  Agent's process (Chat session, a developer's program, any MCP client)
      |   Mila.AI's client
      |   MCP over a named pipe that only the user can open
      v
  Broker (one per user)      policy, approvals, audit trace; runs no tool code
      |   MCP over stdio, one per worker
      v
  Workers                    each tool server in a process of its own:
                             Mila's own tools first; third-party MCP servers
                             once a sandbox holds them (section 6)
```

- **Front: MCP**, revision 2026-07-28 (stateless: every request carries its own `_meta`). Any MCP client can use
  the broker; Mila's applications reach it through Mila.AI's client (section 8).
- **Names:** a tool is shown to the model as `<server>.<tool>`, since MCP tool names are unique only within one
  server.
- **Back: workers.** Mila's own native tools written as servers of their own, and later third-party MCP servers
  and the shell, which wait for the sandbox (section 6). Each is a process of its own.
- **The broker runs as the user, per user, started on demand** (decided 2026-10-24 for the first version;
  section 7 moves it).
- **The compiled-in registry stays** (`Direction.md` 8, decision 6) for tools that need no isolation. A result
  from either returns to the agent's process, which splices it into the live cache as today.
- **Cost:** a warm round trip to a resident process measured about 7 us at the median, against about 14 ms to
  start a process per call (`Notebooks/LocalToolService.md` section 5), and a whole call through the broker -- two
  hops, each message parsed and checked -- about 52 us (section 11). Workers are kept warm; a per-call start
  is used only where section 6.3 calls for a fresh sandbox.

## 4. Policy

The broker is the only holder of the rules. For each (user, agent, tool): **allow**, **ask** or **deny**, with
optional limits -- folders a tool may reach, whether it may use the network, a time and result-size cap.

- **No agent's process can write the policy.** An agent that edits its own harness -- its guardrails, scripts
  and tool-loop plugins are files it rewrites (*Harness^2*, arXiv 2610.14224, section 2.1) -- must find nothing
  of the rules among them. Until section 7's move, the policy is a file in the user's profile, which a program
  running as the user can still change; the broker records any change it did not make.
- **Tool definitions are pinned when approved.** A server's description and schema are prompt text the model
  reads, and MCP says to treat a server's annotations as untrusted. If a server's tool list changes, its
  approvals are withdrawn until the user approves again.
- **Defaults:** a newly registered agent has nothing allowed; every tool is ask, and a headless agent's ask is a
  deny.
- **The audit trace** records each call: user, agent, tool, arguments, decision, who decided, result size,
  duration.

## 5. Approval

When the rule is ask, the broker answers the call with MCP's input-required result; the application shows the
question to the person and retries the call with the answer. The broker decides; the application only presents.

- **As built** (`ToolBroker.Broker`, 2026-10-24): the result carries one form elicitation, `approval`, whose one
  field `decision` offers `allow_once`, `allow_conversation` and `deny`, and a request state. MCP makes that
  state attacker-controlled input, so it is a random token the broker keeps, not data it trusts: bound to the
  agent, the conversation, the tool and its exact arguments, used once, gone after ten minutes. A retry carrying
  no answer is asked again; a decline, a cancel or `deny` is a declined result.
- **Who is asking for what** travels in every request's `_meta`, since MCP keeps no session: the agent under
  `me.toddt.mila/agent`, the conversation under `me.toddt.mila/conversation` (the reverse of the project's own
  domain, as MCP asks of a vendor's keys). A call that names no conversation is not offered
  `allow_conversation`.
- **Nobody to ask:** a client that declares no form elicitation has no person behind it, so its ask is a refusal
  (section 4) -- and MCP forbids sending it an elicitation anyway.

- **Choices:** allow once, allow for this conversation, deny. Making a rule permanent is done in the policy, not
  from a prompt.
- **A denial reaches the model as a tool result** saying the user declined, so it can continue.
- **A stopped reply cancels a pending approval.**
- **What the person is shown** is the tool's name and summary and its arguments in full, before the call runs
  (MCP: show inputs before calling, to catch exfiltration).

Chat's window carries this in its own schema (`ChatDesktop.md` section 4); the console as a prompt line.

## 6. Workers and their limits

**The first broker runs no tool in an operating-system sandbox** (Todd, 2026-10-24). Its tools are Mila's own code
(section 6.1), so what has to be held back is a model steered into misusing tools that behave correctly, and the
broker's own checks hold that. A sandbox exists to hold back code Mila does not control; that code waits for it
(section 6.3). This replaces the decision of the same day that no tool runs unsandboxed, which put an AppContainer
under every worker before the first tool could run (`Notebooks/LocalToolService.md` section 10).

### 6.1 The first tools

Two workers, so the tool that reads the web never sees the user's files and the file tools make no network
connection (decided 2026-10-24, Todd).

| Worker | Reaches | Tools | Default rule |
|---|---|---|---|
| `files` | The project area the person chose for the conversation, and its `.mila` folder; no network | `read_file` (a path, an optional line range), `list_directory`, `find_files` (a name pattern), `search_text` (text or a pattern across the folder) | allow |
| | | `write_file` (create or replace), `edit_file` (replace one exact passage, which must occur once) | ask |
| `web` | Public addresses, by http and https; no files | `fetch_url` (a page as readable text, size-capped) | ask |

**The project area and `.mila`** (Todd, 2026-10-24). The desktop app works in a project area: the folder the
person chose. Inside it, `.mila` is the model's own space, where it creates and keeps files for its own use --
notes, plans, scratch work. Nothing the broker relies on lives there: the policy, the pinned tool definitions and
the audit trace stay where section 4 puts them, because a tool can write anything in `.mila`.

The default rules are those of section 4's "nothing allowed" made usable: they are offered when the person registers
the agent, and the person can change them there.

**Not in the first set:** web search, because every search service needs a key and the first broker holds no
secrets (section 7); and a shell, which runs programs Mila does not control and so waits for the sandbox (6.3).
Both are the next tools after v0.21.0.

### 6.2 What limits a worker

- **Files: the path check.** A `files` worker opens the path it is given, then compares the opened file's final
  path -- after every link, junction and mount point is followed -- with the project area's own final path, and
  closes and refuses anything outside it. A check on the text of the name is not enough on Windows, where one file
  has many names. Before opening, it refuses names that are not plain paths: alternate data streams
  (`name:stream`), device names (`CON`, `NUL`, `COM1`), the `\\?\` and `\\.\` forms, and network (UNC) paths. Short
  8.3 names and differences of case resolve to the same final path and are caught there. A new file is checked
  through its folder, which must itself resolve inside the area.
- **Network: the address check.** The `web` worker resolves each host name itself and refuses private (RFC 1918
  and IPv6 unique-local), loopback, link-local and unspecified addresses, as MCP's guidance against requests to
  internal addresses advises. It checks the address it actually connects to, not a name, so a name that resolves
  differently a second time gains nothing, and it checks again on every redirect.
- **Processes.** Each worker is a process of its own inside a job object with kill-on-close and no breakaway, so a
  fault in a tool never lands in the process holding the model, and closing the broker ends every worker. This is
  separation, not containment: a worker runs as the user and could reach anything the user can, which is why only
  Mila's own code runs there.
- **Environment:** a worker inherits nothing of the user's environment beyond what its server's entry declares.
- **Results are untrusted data**: capped in size before they reach the agent, never interpreted by the broker.
- **Wording.** Nothing shown to the person calls a tool sandboxed. The app says the tools are limited to the
  project folder, and that it asks before writing outside `.mila` and before fetching a page.

**What waits for the sandbox:** a shell, third-party MCP servers, and any tool an agent writes -- a script, a
plugin, a new tool of unknown origin, whatever a library of the agent's adopted it for. Until a platform's sandbox
is built and its escape tests pass, the broker registers none of them there and says so.

### 6.3 The sandbox, next

Filed in `Mila/Issues/Vnext.md`. **Microsoft eXecution Container (MXC)** is the first candidate (Todd,
2026-10-24): one containment policy -- read-write, read-only and denied paths, outbound network, UI -- launched on
Windows through Microsoft's process container, on macOS through Seatbelt and on Linux through bubblewrap, so it
answers section 10.2 as well. What adopting it requires:

- **The escape tests**, each first shown to pass against an unsandboxed worker: a child started through WMI (which
  does not join a job object, on Microsoft's job-object page), the network with nothing granted, a file outside
  the granted folder, and a child process.
- **The connection moves into the broker.** The `web` worker then has no network at all and turns into text what
  the broker fetched under policy, which is NVIDIA OpenShell's design (`Notebooks/LocalToolService.md` 10.3).
- **A fresh sandbox per run** where runs must not see each other: an agent trying two candidate edits (as
  *Harness^2*'s probes do, each in a fresh container) gets a worker created for each run and destroyed after it.
- **A tool an agent wrote** runs only there, recorded as written by that agent and asked about before its first
  run.

## 7. Secrets

**The first broker holds none** (Todd, 2026-10-24). A broker running as the user cannot keep a secret from other
programs running as the user: they can read its memory and decrypt what it stores for that account.

- **The gate for secrets is the broker's move to an account of its own** (a service), which also puts the policy
  out of the user's programs' reach and makes section 2's user check a boundary between accounts. It costs an
  administrator install.
- **A secret never reaches an agent or a model.** The broker gives it only to the worker that uses it, so a
  compromised agent can make the calls it is granted but cannot take a token away.

## 8. Placement

- **The broker:** an application under `Mila/Apps/`, shipped, with a `mila-` command name
  (`Notebooks/MilaCommandLineReview.md`). Working folder `Mila/Apps/Tools/Broker`, in C++23 like Chat and
  `mila-store` (Todd, 2026-10-24), until section 10.3 names it. Built so far, as the library `MilaToolBroker` with
  its tests in `ToolBrokerTests`: the line channel that frames MCP messages on every hop (MCP's stdio framing,
  which its transports page asks a custom transport to reuse); `launchWorker`, the one place a worker is started,
  taking a `WorkerLimits` a sandbox can later enforce whole; the named pipe, its access list naming the current
  user alone, refusing remote clients and refusing to start if its name is taken; and `relayOne`, which checks a
  client's message against MCP's base protocol (`parseClientMessage`), forwards it, and answers for a worker that
  sends a request, the wrong id, something other than JSON-RPC, or nothing. JSON is the vendored header Mila
  itself wraps (Todd, 2026-10-24). Then `Broker` (2026-10-24): `tools/list` over every worker added to it, each
  server's list read once when it is added; `tools/call` routed to its server under the tool's own name; the
  rules of section 4 as a `Policy` held in memory; and section 5's approvals.
- **The client:** in Mila.AI (`Mila/AI/`), beside where the agent core will be (`Direction.md` 3.2). **Mila.AI
  starts in v0.21.0 with the client as its only content** (Todd, 2026-10-24), so it is written once, where v0.22.0
  needs it, and no application holds a client of its own (`Direction.md` 3.3). It needs nothing from `Mila/Src`: a
  pipe, JSON and MCP. Built 2026-10-24 as the target `MilaAI` (`Mila::AI`), module `Mila.AI`, namespace
  `Mila::AI`: `BrokerClient` lists the broker's tools and calls one, turning an input-required result into an
  `ApprovalRequest` for the application's `Approver` and retrying with the answer. Its interface holds no JSON
  type; arguments and results are text. Its gate passes in `MilaAITests`: calls through a real broker, its pipe
  and echo workers, with no application.
- **Mila's own tool workers:** under the broker's folder, one program per server.

## 9. Phases

Each phase's gate is written before its run.

1. **Round-trip cost** -- done, 2026-10-23 (`Notebooks/LocalToolService.md` section 5).
2. **The path and address checks on Windows, first** (section 6.2): a test for every way a name can lead out of
   the project area -- `..`, a link, a junction, a mount point, a short 8.3 name, a difference of case, an
   alternate data stream, a device name, a `\\?\` or `\\.\` form, a network path -- and for every refused address
   kind, by name, by literal address and through a redirect. Each test is first shown to reach its file or address
   with the check removed, then shown refused with it.
3. **End to end** -- done, 2026-10-24: client to broker to a worker, with MCP framing and JSON on both hops, a
   median of about 52 us against the gate's 100 (`Notebooks/LocalToolService.md` section 11).
4. **The broker and the two workers of section 6.1**, with section 4's rules and section 5's approvals. The rules
   and approvals are built (2026-10-24); the workers wait on phase 2.
5. **The client in Mila.AI, and Chat on it:** the console and the window ask through section 5. The client is
   built (2026-10-24), and both front ends ask before a tool runs; neither reaches the broker yet.
6. **The sandbox** (section 6.3), then the shell and third-party servers; after v0.21.0.

## 10. Open decisions

1. **Giving a sandboxed tool the user's files** -- deferred with the sandbox (6.3). Granting an AppContainer a
   folder means adding an entry to that folder's access-control list, which Windows copies onto every file
   beneath it and which has to be taken back; MXC's own process container leaves the grant to the operating
   system instead. Measured when the sandbox is (`Notebooks/LocalToolService.md` 10.1).
2. **Linux and macOS sandboxing:** MXC's Seatbelt and bubblewrap backends are the first candidates (6.3). Not
   measured.
3. **The name** of the broker and its command.
4. **The policy's format** and how the user edits it: a file, a command, a page in Chat's window.
5. **Registering an agent:** how the user does it once per application, and what a developer's program shows
   the first time it calls.
6. **Writes inside `.mila`.** Proposed: allow by default, since asking each time the model writes its own notes
   wears the person out, while writes elsewhere in the project area stay ask. Not decided.
