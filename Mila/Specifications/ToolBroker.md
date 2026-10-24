# Tool Broker

One local service that executes the tool calls of every agent on a machine -- Mila's own applications and other
people's -- under one set of rules, each tool in a sandbox of its own.

**Status:** draft, 2026-10-24. No code. The evidence behind it, and the reasoning for each decision, is
`Notebooks/LocalToolService.md`. **Ships in v0.21.0 on Windows** (Todd, 2026-10-24), behind the desktop app's tool
calling (`ChatDesktop.md`): the ROADMAP theme *The Desktop Agent*. Open decisions are in section 10.

---

## 1. What this is

Today each program that hosts a Mila model executes its model's tool calls itself: Chat through its own handler
map (`registerTool`, `Mila/Apps/Chat/Src/Chat.ixx`), a developer's program through whatever it writes. Nothing
holds the rules in one place, nothing restricts what a tool can reach, and no Chat asks before a tool runs.

The broker is a separate process that holds the rules and runs the tools. An agent asks it to call a tool; the
broker decides whether that agent, acting for that user, may; and if so, a sandboxed worker runs the call and
the result comes back.

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
  Workers                    each tool server in its own sandbox:
                             third-party MCP servers, and Mila's own tools
```

- **Front: MCP**, revision 2026-07-28 (stateless: every request carries its own `_meta`). Any MCP client can use
  the broker; Mila's applications reach it through Mila.AI's client (section 8).
- **Names:** a tool is shown to the model as `<server>.<tool>`, since MCP tool names are unique only within one
  server.
- **Back: workers.** Third-party MCP servers, and Mila's own native tools (file read, grep, patch apply, the
  shell) written as servers of their own. Each is a sandboxed process (section 6).
- **The broker runs as the user, per user, started on demand** (decided 2026-10-24 for the first version;
  section 7 moves it).
- **The compiled-in registry stays** (`Direction.md` 8, decision 6) for tools that need no isolation. A result
  from either returns to the agent's process, which splices it into the live cache as today.
- **Cost:** a warm round trip to a resident process measured about 7 us at the median, against about 14 ms to
  start a process per call (`Notebooks/LocalToolService.md` section 5). Workers are kept warm; a per-call start
  is used only where section 6 calls for a fresh sandbox.

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

- **Choices:** allow once, allow for this conversation, deny. Making a rule permanent is done in the policy, not
  from a prompt.
- **A denial reaches the model as a tool result** saying the user declined, so it can continue.
- **A stopped reply cancels a pending approval.**
- **What the person is shown** is the tool's name and summary and its arguments in full, before the call runs
  (MCP: show inputs before calling, to catch exfiltration).

Chat's window carries this in its own schema (`ChatDesktop.md` section 4); the console as a prompt line.

## 6. Workers and sandboxes

- **Windows:** each worker in an AppContainer (files and registry only where granted; network only by
  capability; isolated kernel objects and windows), inside a job object with kill-on-close, memory and CPU caps,
  and no breakaway. A child started through WMI does not join the job, so phase 3 tests that route.
- **Linux and macOS:** not yet read (section 10). **Windows first** (Todd, 2026-10-24): until a platform's sandbox
  is built and its escape tests pass, the broker runs no tool there and says so; no tool runs unsandboxed.
- **Environment:** a worker inherits nothing of the user's environment beyond what its server's entry declares.
- **A fresh sandbox per run** where runs must not see each other: an agent trying two candidate edits (as
  *Harness^2*'s probes do, each in a fresh container) gets a worker created for each run and destroyed after it.
- **A tool an agent wrote** -- a script, a plugin -- is a new tool of unknown origin: it runs only as a worker,
  is recorded as written by that agent, and is asked about before its first run, whatever any library of the
  agent's adopted it for.
- **Results are untrusted data**: capped in size before they reach the agent, never interpreted by the broker.

### 6.1 The first tools

Two workers, each in its own sandbox, so the tool that reads the web never sees the user's files and the file tool
has no network (decided 2026-10-24, Todd).

| Worker | Reaches | Tools | Default rule |
|---|---|---|---|
| `files` | The one folder the person chose for the conversation; no network | `read_file` (a path, an optional line range), `list_directory`, `find_files` (a name pattern), `search_text` (text or a pattern across the folder) | allow |
| | | `write_file` (create or replace), `edit_file` (replace one exact passage, which must occur once) | ask |
| `web` | The internet, by http and https; no files | `fetch_url` (a page as readable text, size-capped; private, loopback and link-local addresses refused, as MCP's guidance against requests to internal addresses advises) | ask |

The default rules are those of section 4's "nothing allowed" made usable: they are offered when the person registers
the agent, and the person can change them there.

**Not in the first set:** web search, because every search service needs a key and the first broker holds no
secrets (section 7); and a shell, the largest risk, whose behaviour inside an AppContainer has not been tried. Both
are the next tools after v0.21.0.

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
  (`Notebooks/MilaCommandLineReview.md`).
- **The client:** in Mila.AI (`Mila/AI/`), beside where the agent core will be (`Direction.md` 3.2). **Mila.AI
  starts in v0.21.0 with the client as its only content** (Todd, 2026-10-24), so it is written once, where v0.22.0
  needs it, and no application holds a client of its own (`Direction.md` 3.3). It needs nothing from `Mila/Src`: a
  pipe, JSON and MCP.
- **Mila's own tool workers:** under the broker's folder, one program per server.

## 9. Phases

Each phase's gate is written before its run.

1. **Round-trip cost** -- done, 2026-10-23 (`Notebooks/LocalToolService.md` section 5).
2. **The folder and the escapes on Windows, first:** granting a worker's AppContainer one folder and taking it
   back (section 10.1), and the escape tests -- a WMI-started child, network with no capability, a file outside the
   granted folder, a child process. Each escape must fail, and each test is first shown to pass against an
   unsandboxed worker. If the folder grant does not work cleanly, the plan changes here, before anything is built
   on it.
3. **End to end:** client to broker to a worker in an AppContainer, with MCP framing and JSON on both hops.
   Gate: median under 100 us.
4. **The broker and the two workers of section 6.1**, with section 4's rules and section 5's approvals.
5. **The client in Mila.AI, and Chat on it:** the console and the window ask through section 5.
6. **Linux**, after its sandboxing is read; after v0.21.0.

## 10. Open decisions

1. **Giving a sandboxed tool the user's files.** Granting an AppContainer a project folder changes that folder's
   permissions and has to be undone; less-privileged AppContainers restrict further. Whether this is usable for
   coding work decides whether the Windows sandbox is (`Notebooks/LocalToolService.md` 6.3).
2. **Linux and macOS sandboxing:** bubblewrap, Landlock and seccomp on Linux; macOS's own. Not read.
3. **The name** of the broker and its command.
4. **The policy's format** and how the user edits it: a file, a command, a page in Chat's window.
5. **Registering an agent:** how the user does it once per application, and what a developer's program shows
   the first time it calls.
