# Mila, the App

Mila, the desktop app (`Mila/Apps/Desktop/`): an agent a person puts to work, organised around the work they
came to do rather than around models (section 3). It runs over the same session as Chat's console. The
session runs as its own C++ process; the window is a client of it, in whatever language suits the window. The
console stays first-class on every platform. Named `ChatDesktop.md` until 2026-10-26.

**Status:** draft, 2026-10-20. Phase 1 (the page and the Electron window, against a mock session) is
built; phase 2 (the session split, section 7) and phase 3 (the session process, section 6.1) are built,
2026-10-24, and the window runs over the real session (phase 4, section 8). The app's name and audience are
decided (section 9, 2026-10-24); its direction, activities over models, on 2026-10-26 (section 9.7). The
layout of section 3 is that direction; the page as built is still the conversation-and-models layout it
replaces (phase 6, section 8). Open decisions are in section 9.

---

## 1. What this is

Chat is a session and a console front end over it: `ChatSession` (`Chat.Session.ixx`) holds the model
resolution, the deployment request and the model as a `Mila::AI::Agent`, which runs the turn, its tool calls and
the channel routing (`+98`), and the console
(`Chat.ConsoleFrontEnd.ixx`) draws its events through `ConsoleRenderer` and reads `std::cin` (section 7). A
window is a second front end over that same session. It exists because the things a person now does with a local model are window-shaped:

- **Media.** An image or an audio clip attached by drag-and-drop, shown in the transcript, and refused by
  name for a model that has no such modality (`Gemma4Modality.md` 5.7). A terminal can take a path; it
  cannot show the picture back.
- **Conversations.** A sidebar of past conversations, reopened where they were left. Chat has no
  conversation store today: every session starts empty.
- **The showcase for end users.** Mila is the brand; the desktop app is called Mila and is where a person
  who runs local models sees what Mila can do -- an agent with tool calling, with the conversation as its
  face. The library's audience is the developer; the app's is the person running it (Todd, 2026-10-24,
  section 9). Everything the app shows a person is end-user prose (CLAUDE.md, *End-User Prose*).

**The person chooses the work; Mila chooses the intelligence** (Todd, 2026-10-26). The app is not a place to
configure and run models -- a catalogue of models, each with its sampling, context and quantization knobs, is
the layout of LM Studio and its peers, and Mila's case against them is the quality of the agent, not the breadth
of the catalogue. A person starts from what they want done; the model, its deployment and the agent's limits
are Mila's choice, made from what Mila has measured, visible on request and capped where the person wants a
boundary. The comparison is cuBLASLt's: the caller states the operation and its constraints, the library picks
the implementation. Every element on the main path earns its place by helping the person get the work done,
showing what Mila is doing, or setting a boundary that matters.

**An app adds no capability to the library** (CLAUDE.md). The window shows what the library and the
session already do. Video, which no Mila model takes, has no place in it. Nor does an activity whose tools or
agent behaviour the library and the app do not yet have: a card is shown when its work can be done (section 3).

## 2. Architecture

```
  +----------------------+     +---------------------------------------------+
  |  Console front end   |     |  Window: Electron (section 6.2)             |
  |  ConsoleRenderer,    |     |  page Ui/  <-- bridge -->  window process   |
  |  stdin               |     +----------------------+----------------------+
  +----------+-----------+                            |  JSON lines on stdio
             |                                        |  (section 4)
             |                 +----------------------+----------------------+
             |                 |  Session process: the JSON-lines front end  |
             |                 +----------------------+----------------------+
             |  events out / requests in               |
             +--------------------+--------------------+
                                  |
                   +--------------+---------------+
                   |  Chat session                |
                   |  (Chat.Session.ixx)          |
                   +--------------+---------------+
                                  |
                                Mila
```

- **The line is presentation versus session.** It is the line v0.22.0 draws when Chat is rebuilt on
  `Mila::AI` ("Chat keeps terminal rendering and the approval gate", `Vnext.md`): what is beneath it moves
  then, and no front end does.
- **The session is its own process; the window is a client** (Todd, 2026-10-20). The C++ never draws a
  window: the session process is the same small program on every platform Chat builds on, and the window
  can be written in JavaScript, .NET, Python, Rust or anything else that can start a process and read
  its output. A fault in either does not take down the other, so a window crash does not unload the model.
  Mila's window is Electron (section 6.2); other clients are welcome on the same schema.
- **Formatting is presentation.** The session emits channel-routed *raw* text; routing a reply to its
  channels is the session's, through the template's reader in the library (`Conversation::ReplyReader`). The
  console front end
  keeps `IncrementalRichFormatter` and `RichText` (markdown and LaTeX to terminal Unicode, word wrap); the
  window renders markdown itself, in the page. Each front end keeps its own streamed-equals-buffered
  check.
- **The page is HTML, CSS and JavaScript.** All layout, rendering and interaction live in `Ui/`; the
  window process only hosts it, starts the session process, and relays messages.
- **The session runs on the library directly; not through the inference server.** The server's wire is
  the OpenAI and Anthropic APIs, stateless per request, and has no model install, load or remove, no
  progress, no context rebuild and no session settings; giving it those would make it an app back end,
  which it is not. It would also put a Python runtime and a local port in an end-user install. The two
  are servers with two jobs: the inference server is the industry wire for other tools, the session
  process is Mila's own stateful protocol for Mila's own front ends.
- **"In process" is unchanged.** `Direction.md`'s in-process property is about a developer harnessing
  Mila, which stays the library and its bindings. The tool loop and the KV splice still run inside the
  session process; only presentation crosses the process boundary.
- **The window never reaches the library.** It speaks only the schema in section 4, so a third copy of
  the family-specific code is never written. The session process links Chat's today, and with it Chat's
  family-specific types; that goes in v0.21.0 (`BACKLOG.md` *The Desktop Agent*, `Direction.md` 4.3).

## 3. Layout

A sidebar on the left; on the right, the start of new work, or the work under way. Organised around activities,
not models (section 1; decided 2026-10-26, section 9.7).

### 3.1 Activities

An activity is what a person came to do. It is the app's vocabulary, and it decides what the app gives the agent
for that work: which of the app's tools, the default rules for them (`ToolCalling.md` 3.3), the guidance in the
system prompt, and the agent's limits. It does not name a model. A conversation keeps the activity it started in.

| Card | What it is for | Shown when |
|---|---|---|
| **Ask & Explore** | Questions, explanations, ideas, working a problem through | Now: it needs no tools |
| **Build & Code** | Writing, debugging and improving software in a folder the person chooses | The app's file tools exist (BACKLOG, "A model has no tools to read or edit files") and run through their grants ("Nothing keeps a tool inside the folder it was given") |
| **Research & Analyze** | Investigating a question across sources, comparing evidence | The app's web-reading tool exists: shown since 2026-10-26, reading the pages the person names or the model knows, through its grant: public addresses only (6.1). Web search with Tavily (Todd, 2026-10-26) once the person has given their own key in Settings (3.7). Analysing data by running code waits for a sandbox (`Vnext.md`) |
| **Get Things Done** | An outcome to reach in steps: a plan the person approves before it runs, progress, a deliverable | Shown since 2026-10-26 (Todd), in the app's form: in a folder, with the file tools and web reading, and a plan the model proposes (`propose_plan`, always asked about) and marks step by step (`update_step`, allowed by rule), shown as a checklist above the box and kept with the conversation. A plan the person edits in place, and the task itself run and checked by `Mila::AI` rather than by the model's own discipline, remain the goal (9.7) |

A card whose work cannot yet be done is not shown, rather than shown disabled. Nothing guesses the activity from
what a person types: guessing is a model call before the first token, or a guess. The box on the home screen
starts *Ask & Explore*; a card is the way into the others. Changing an activity inside a conversation changes the
agent's tools and so its system prompt, which costs a prefill of the whole conversation; whether to offer it is
open (section 9.7).

### 3.2 The model is Mila's choice

What an activity needs -- a model whose manifest carries `agentic` (`ModelDistribution.md`), the context it works in, a modality -- is the
library's vocabulary; which installed model meets it on this card, deployed how, is a question the library answers,
not the app (the no-new-capability rule). That query does not exist yet: it would rank the installed models on
what Mila has measured, never on a family name (BACKLOG, "What a model can do is inferred from its family name"),
and plan each on the device (`Deployment.md`). Its nearest anchor is v0.22.0's use case (`ROADMAP.md`, *Mila::AI*:
"a named selection of features over the same request", its defaults from measurement), which deploys one model for
a use case; choosing which installed model serves it goes one step past that narrative. It is a proposal for
`Mila::AI`, not agreed. Until it exists, the app uses the model the session loads, as today.

What has been measured today is one judgement, agentic or not, plus whether a model fits the card; on a 16 GB card
every activity would choose the same model. That is expected: what differs between activities is what the agent
is given, not who runs it.

**First run.** A fresh store has no model (CLAUDE.md), so the home screen with nothing installed recommends one
model for this card -- the measured agentic model that fits it, from the published list -- with one *Install* and
its progress, not a catalogue.

### 3.3 Sidebar

- *New* at the top (Ctrl+N): the home screen.
- Conversations, newest first, grouped *Today*, *Yesterday*, *Previous 7 days*, *Older*. Each row shows
  its title and its activity; hover shows the model it was held with. Rename and delete from a row's menu;
  delete asks first.
- At the foot: *Settings*. The models view is reached from Settings, off the main path.
- Collapsible; collapsed below 900 px of window width.

### 3.4 Home screen

- One line asking what the person wants to do, and one box under it; sending from it starts *Ask & Explore*.
- The cards of 3.1 that can be shown, each its name and one line of what it is for.
- Nothing else: no model, no settings. With no model installed, the first-run recommendation of 3.2 takes the
  cards' place until a model is installed.

### 3.5 Main pane

- *Header:* the conversation's activity, and as quiet text what Mila chose for it -- the model, the context window
  -- which opens on request to the detail: the model and its package, context (with "auto" when it was
  measured, not chosen), thinking, the tools the activity gives the agent and their rules. Read-only there; a
  change is made under the limits (below) or, for the model, in the models view. With no model loaded, the
  header says so and the way out.
- *Transcript:*
  - User turns, right-aligned, with their attachments as thumbnails (image) or a small player (audio).
  - Assistant turns, full width. The reasoning channel is a collapsed block above the answer ("Thought
    for 412 tokens"), expandable, streamed live when open. Tool calls are compact cards in order of
    occurrence: the tool's name and summary, the result expandable. A call waiting for the person's
    approval is open, shows its arguments in full and "Waiting for you", and offers *Allow once*, *Allow in
    this conversation* and *Deny*; a declined call says "Declined".
  - Markdown rendered: headings, emphasis, lists, tables, links, inline code, fenced code with a
    language label and a copy button. Math is shown as its source in phase 1 (section 9).
  - Below each finished reply: copy, and a quiet stats line on hover (prefill ms, tok/s, tokens, rounds).
    A reply that ended on the token cap or the context bound says so in the reply, not only in stats.
- *Log drawer:* at detail `all` only, a collapsed "Log" bar between the transcript and the composer, with the
  count of lines; opened, the session's `log` lines in monospace, newest at the bottom.
- *Composer:* a multi-line box (Enter sends, Shift+Enter is a newline), an attach button, drop anywhere on
  the pane to attach, attachments as removable chips above the box. While a reply is generating, *Send*
  becomes *Stop* (Esc). Attach is disabled with a tooltip naming why for a model without that modality.

### 3.6 Limits

The person's caps on what Mila chose, in a small panel from the header. Every one defaults to Mila's choice, and a
cap is a hard bound, not a preference: a reply that meets one stops and says so, with what it had done.

- *Ask before* -- which of the activity's tools run without asking: the rules of `ToolCalling.md` 3.3, the
  activity's defaults shown. Being allowed to proceed never widens what a tool may reach; that is its grant.
- *Time* and *length* of a reply -- `AgentLimits` (`+98`).
- *Effort* -- only where the model's template has levels, from its default; a model without them shows none,
  since no layer invents a reasoning mode a model's template does not have.

There is no cap on agents in parallel: `Mila::AI` runs one.

### 3.7 Settings and the models view

Off the main path, for the person who wants them.

**Web search** (in Settings, 2026-10-26): the person's own Tavily key, pasted once and shown only as set, with
*Remove key*. The session keeps it in `search.json` beside the conversations folder, in plain text until the
operating system's credential store holds it, and sends it only to Tavily, in a basic search (one credit) of at most
ten results (`Desktop.WebSearch.ixx`). With a key, Research & Analyze and Get Things Done are offered `web_search`,
and every search is asked about with its words shown, since they leave the machine; without one, the tool is not
offered. Tavily's search is a POST, which Mila's HTTP transport sends since that day (`HttpClient.md` 4.1). Not yet
read: Tavily's terms on keeping results, which a saved conversation does.

**Settings** (a side panel): the limits' defaults for new conversations, and under *Advanced* the session's
knobs -- sampling (temperature, top_k, top_p, presence_penalty, each with *model default*), thinking on/off,
context (a number or *auto*), detail (off, thoughts, all), seed. Each reflects `session.state`; changing one
sends `settings.set`. The defaults are the publisher's (CLAUDE.md, *Measurement*), so a person who never opens
*Advanced* runs the model as it was measured.

**Models view** (a pane over the main pane, from Settings): installed models with size and quantization, *Load*
(with the quantization choice for unquantized weights: none, fp8, fp4, q4_0), *Remove* (asks first);
*Available to install* from the published list, *Install* with progress. A load or install that is refused
shows the refusal's own text at the top of the view, which covers the transcript, until the next load,
install or remove. Once the selection of 3.2 exists, *Load* here is an override of Mila's choice, and says so.

**The schema** (section 4) carries the activity and its folder: `session.state` lists the activities the session
can start, `conversation.new` names one, and `conversations.list`, `conversation.loaded` and the stored conversation
keep it. The folder is chosen in the window's own picker (`folder.choose`), never typed.

**Look:** light and dark follow the system. The palette and type follow the website
(`Web/layouts/_default/baseof.html`), quieter. No animation beyond the streaming caret and a progress
bar.

## 4. Message schema

JSON objects, one per message, both directions. Every message carries `"v": 1` and `"type"`. Unknown
types are ignored by both sides, and logged. Strings are UTF-8. Ids are strings chosen by the side that
creates the thing (the session for turns and conversations, the page for attachments).

**This schema is a contract** once a client in another language targets it: a change that an existing
client would misread steps `"v"`.

**Streams.** A session process holds one loaded model and one or more streams over it -- a stream being
one conversation's state on that model. Every `turn.*` message, in both directions, carries a `stream_id`;
`session.state` lists the streams. Chat's window uses one stream. The field is there from `"v": 1` so
that a process running several streams over one set of weights is not a version step.

**Transport.** Between the window process and the session process: one JSON message per line (newline
delimited; JSON escapes any newline inside a string), requests on the session process's standard input
and events on its standard output, both UTF-8. Standard error carries the session's and the library's
log records -- the library writes its records to `std::cout`, so the session process points `std::cout`
at standard error and writes the protocol to the real standard output, as `mila-chat -p` already does
for its answer (`Chat::runOnce`). End of input ends the session process. Between the page and the
window process, the window's own bridge (section 6.2).

The window relays every message unchanged in both directions, except the three it handles itself:
`link.open`, `folder.choose`, and resolving a dropped file to a path (section 6.4).

### 4.1 Session to page

| type | fields | meaning |
|---|---|---|
| `session.state` | `model` (name or null), `no_model_reason` (string or null), `context_length`, `context_automatic`, `thinking_capable`, `thinking` (`"off"`, `"on"` or an effort name), `effort` (1-5 or null), `effort_levels` (bool), `detail` (`"off"`, `"thoughts"`, `"all"`), `sampling` ({`temperature`, `top_k`, `top_p`, `presence_penalty`}, each `{value, is_model_default}`), `modalities` (subset of `["image","audio"]`), `activities` (the activities this session can start, in card order, a subset of `["ask","build","research","tasks"]`), `activity` (the conversation's), `folder` (the conversation's project folder, or null), `plan` (Get Things Done's approved plan, [{`text`, `status`: `"pending"`, `"in_progress"`, `"done"` or `"skipped"`}], or null), `web_search` (bool: a search key is set; the key itself is never sent), `busy` (bool), `streams` ([{`stream_id`, `parent_stream_id` (or null), `conversation_id`, `busy`}]) | The whole session state. Sent after every change, never as a delta. |
| `models.list` | `installed` ([{`name`, `size_bytes`, `quantization`, `family`}]), `published` ([{`name`, `size_bytes`, `quantization`, `installed`}] or null when not fetched) | Reply to `models.refresh`, and after install or remove. |
| `progress` | `task` (`"load"`, `"install"`, `"rebuild"`), `label`, `done`, `total` (both null when unknown) | A long task's progress; a final one has `done == total`. |
| `turn.begin` | `turn_id`, `conversation_id` | A reply starts. |
| `turn.text` | `turn_id`, `round`, `channel` (`"thought"` or `"answer"`), `text` | Raw text to append to that channel. Gemma streams many; Llama and Qwen send one per round when it is complete. The page must render both. |
| `turn.thought_break` | `turn_id`, `round` | The reasoning channel resumed after a tool call; start a new thought block. |
| `turn.tool_call` | `turn_id`, `round`, `call_id`, `name`, `summary`, `arguments` (JSON string) | The model called a tool. |
| `turn.tool_approval` | `turn_id`, `call_id`, `choices` (subset of `["allow_once", "allow_conversation", "deny"]`, in that order) | The call named by `call_id`, already sent as `turn.tool_call`, waits for the person's decision (section 4.5). |
| `turn.tool_result` | `turn_id`, `call_id`, `result`, `is_error` | The tool's result, fed back to the model. |
| `turn.round_end` | `turn_id`, `round`, `prefill_ms`, `tokens_per_second`, `tokens`, `thought_tokens` (those of `tokens` the reasoning channel took; 0 when it did not open), `finish` (`turn.end`'s values, or `"tool_call"` for a round that ended in one) | One generate call ended. `thought_tokens` labels the round's thought block ("Thought for 412 tokens"), which the page cannot count from text. |
| `turn.end` | `turn_id`, `finish` (`"stop"`, `"length"`, `"context"`, `"cancelled"`, `"error"`), `answer` (the parsed final answer) | The reply ended. `answer` is what the conversation keeps. |
| `conversations.list` | `items` ([{`id`, `title`, `updated` (ISO 8601), `model`, `activity`, `folder`}]) | After any change to the store. A conversation stored before activities is `"ask"` with no folder. |
| `conversation.loaded` | `id`, `title`, `model`, `activity`, `folder`, `messages` (section 4.3) | The conversation now in the main pane. |
| `attachment.accepted` | `attachment_id`, `kind`, `name`, `size_bytes` | The session has read and decoded the file. |
| `attachment.refused` | `attachment_id`, `reason` | Not found, not decodable, too large, or no such modality on this model. |
| `info` | `text` | A sentence the console would print with `printInfo`. Shown as a quiet line in the transcript. |
| `error` | `text`, `during` (`"load"`, `"turn"`, `"install"`, `"settings"`, `"conversation"`, or null) | Shown where it happened, never as a modal. |
| `folder.chosen` | `path` (or null when the person cancelled) | From the window, not the session: the answer to `folder.choose`. |
| `log` | `level`, `text` | Only at detail `all`; shown in a collapsible log drawer. |

### 4.2 Page to session

| type | fields | meaning |
|---|---|---|
| `ready` | -- | The page has loaded; the session replies with `session.state`, `conversations.list` and `models.list`. |
| `turn.send` | `text`, `attachment_ids` | A user turn. Refused with `error` while `busy`. |
| `turn.stop` | -- | Stop the reply in flight; it ends with `finish: "cancelled"`. A call waiting for approval is not run. |
| `turn.tool_decision` | `turn_id`, `call_id`, `decision` (one of the approval's `choices`) | The person's answer to a `turn.tool_approval`. One for a call not waiting is ignored. |
| `conversation.new` | `activity` (`"ask"` when absent), `folder` (for `"build"` and `"tasks"`) | Clear the session's history and start a new conversation in that activity: its system prompt and its tools (section 3.1). A `"build"` or `"tasks"` without a folder that exists, or an activity not in `session.state`'s `activities`, is refused with `error` (`during: "conversation"`) and the conversation is unchanged. |
| `folder.choose` | -- | Handled by the window, which shows the system's folder picker and answers `folder.chosen`. |
| `conversation.open` | `id` | Replace the session's history with that conversation's. |
| `conversation.rename` | `id`, `title` | |
| `conversation.delete` | `id` | |
| `models.refresh` | `online` (bool) | |
| `model.load` | `name`, `quantization` (`"none"`, `"fp8"`, `"fp4"`, `"q4_0"` or null) | Clears the history, as `/model load` does. |
| `model.install` | `name` | |
| `model.remove` | `name` | |
| `settings.set` | one of: `context` (n or `"auto"`), `thinking` (bool), `effort` (1-5), `detail`, `sampling` ({`key`, `value` or `"model"`}), `seed`, `search_key` (the person's Tavily key, or null to remove it) | One change per message; the reply is `session.state`, or `error` with `during: "settings"`. |
| `attachment.add` | `attachment_id`, `kind`, `name`, `path` | The file on this machine; the session reads it (section 6.4). |
| `attachment.remove` | `attachment_id` | |
| `link.open` | `url` | Handled by the window, which opens it in the system browser. The page never navigates. |

### 4.3 Conversation messages

`[{ "role": "user" | "assistant" | "tool", "content": string, "thought": string | null,
"tool_calls": [{ "call_id", "name", "summary", "arguments", "decision", "result", "is_error" }] | null, "attachments": [{ "kind", "name" }] | null,
"stats": { "prefill_ms", "tokens_per_second", "tokens", "rounds", "finish" } | null }]`

The same shape is the stored conversation (section 6.3). An attachment's bytes are not stored in phase 1;
a reopened conversation shows the attachment's name only. A call keeps what its card showed live, so a
reopened card reads the same: the `summary` of its `turn.tool_call`, the person's `decision` (null when the
call was not asked about), and the `result` and `is_error` of its `turn.tool_result`. A call the person
stopped before it ran has a `result` of null and is shown as not run. (Phase 4: the real session stored
none of these, and a reopened card showed the tool's name twice, lost "Declined", and dropped a stopped call.)

### 4.4 Commands

Every console command has its message: `/clear` is `conversation.new`, `/model load` is `model.load`,
`/context`, `/thinking`, `/effort`, `/set`, `/verbose` and `/seed` are `settings.set`, `/stats` is the stats
line, `/model list` is `models.refresh`. The composer does not parse slash commands.

### 4.5 Approving a tool call

The session asks; the page only presents. Who decides is `Mila::AI`'s permissions (`ToolCalling.md` 3.3), whose
hook for asking the person these messages carry; until the session is built on them, its own rules decide, and a
tool the person has not allowed is asked about.

- After `turn.tool_call`, the session sends `turn.tool_approval` and the reply waits; `busy` stays true.
- `allow_once` runs this call. `allow_conversation` runs it, and later calls of the same tool in this
  conversation run without asking; `conversation.new`, `conversation.open` and `model.load` end that.
- `deny` does not run the call. The session sends `turn.tool_result` with `is_error: true` and a result saying
  the person declined, which is what the model reads, and the reply continues.
- `turn.stop` while a call waits ends the reply as `cancelled` with no result for the call.
- A stored conversation keeps the declined call's result text, as for any call.

## 5. The page (`Mila/Apps/Desktop/Ui/`)

- **Plain HTML, CSS and JavaScript modules. No build step, no framework, no network.** The page loads
  only its own files: a `Content-Security-Policy` of `default-src 'self'; img-src 'self' blob:` in
  `index.html`, no CDN, no web fonts fetched. `blob:` is the attachment previews of section 6.4, object
  URLs over files the user picked; `media-src` gains it when audio has a preview. It ships as files
  inside the window's package.
- **The markdown renderer is hand-written** for the subset in section 3, in phase 1. A vendored library
  would need its licence read at its source and a `NOTICE.md` entry, as Chat's image decoder does
  (`Gemma4Modality.md` 8, decision 3), which is a decision for a local change, not phase 1.
- **The page's text is end-user prose** (CLAUDE.md, *End-User Prose*): no term that exists only inside
  Mila -- not "session process", "stream", "schema", "channel" or "front end" -- in anything the page
  shows. "Thinking", "context window", "model", "install" are the reader's words, as the console already
  uses them (`Chat.ConsoleFrontEnd.ixx`, `printHelp`).
- **Code follows CLAUDE.md's naming rule** in JavaScript too: no abbreviations in identifiers
  (`message`, not `msg`; `conversation`, not `conv`).
- **Realistic content in the mock:** the weather tool's definition is
  `Mila/Apps/Chat/Prompts/tools-weather.json`, and the console shows a tool call through
  `ConsoleRenderer::printToolCall` (`Chat.Renderer.ixx`), which is what `summary` carries.
- **Model text is untrusted.** It never reaches `innerHTML` unescaped: raw HTML in a reply renders as
  text, links are `link.open`, images in markdown are not fetched.
- **One transport interface** (`transport.js`): `send( message )`, `onMessage( handler )`, and
  `pathForFile( file )`, which resolves a dropped or picked `File` to its path on disk, or to null where
  the window cannot. Two implementations in phase 1: Electron's bridge (section 6.2), exposed by its
  preload script as `window.mila`, and a mock, chosen when `window.mila` is absent. Nothing outside
  `transport.js` names Electron. The page targets Chromium only: the window is Electron everywhere.
- **State is reduced from messages**, in a module with no DOM access (`state.js`: `reduce( state,
  message ) -> state`), so the page's behaviour is testable without a browser.
- **The mock** (`mock/`) plays scripted scenarios over the schema with realistic timing, and answers
  requests as the session would. It is one implementation used two ways, so it has no DOM access: the
  page's mock transport calls it directly in a browser, and a Node wrapper serves it on stdio as a mock
  session process speaking JSON lines (section 4, *Transport*), which the window starts in place of the
  real one (section 6.2). Scenarios, each selectable from a dev-only menu (`?scenario=` in the URL, or
  the mock session's `--scenario` argument):
  1. Gemma reply streamed: thinking, a `get_weather` tool call the rules allow and its result, more
     thinking, an answer with a table and a code block.
  2. Llama reply buffered: one `turn.text` per round, at the end.
  3. A reply stopped by the user mid-stream.
  4. A reply ending on `length`, and one on `context`.
  5. No model loaded, then a load with progress, then a load refused with the refusal's text.
  6. An image attached to a model with the modality, and refused on one without.
  7. Five conversations across the date groups; open, rename, delete.
  8. Install with progress from the published list; remove.
  9. Detail `all`: log lines in the drawer.
  10. A reply of 20,000 tokens, to see that appending stays smooth.
  11. A tool call that asks first (section 4.5): allowed, the reply continues with the result; denied, the
      model says it could not check and answers without it. A second message in the same conversation
      after *Allow in this conversation* runs the call without asking.
- **Tests:** `node --test` over `state.js`, the markdown renderer and the mock's scenario scripts. No
  committed test dependency beyond Node itself. The visual checks of phase 1's gate may use whatever
  headless browser the building session has, with nothing of it committed.

## 6. The session process and the window

### 6.1 The session process (phase 3, local)

A C++23 program in `Mila/Apps/Desktop/Session/`, linking Chat's session library (`MilaChatSession`, section 7),
built wherever Chat builds -- Windows and Linux today, macOS once a Metal backend exists
(`Future.md`, "Compute backends beyond CUDA") -- as a front end over the phase 2 interface: it reads
section 4.2's requests from standard input and writes section 4.1's events to standard output (section 4,
*Transport*). One loaded model per process, and one stream for Chat's window (section 4, *Streams*).
Generation runs off the thread that reads requests, so
`turn.stop` is read while a reply is streaming. It never draws, and it is testable headless: a script
writing request lines and reading event lines is a complete client.

It stays thin: it translates between the schema and the session and holds no logic of its own, so that
when Chat is rebuilt on `Mila::AI` in v0.22.0 the process becomes a thin layer over the `Mila::AI` API
(Todd, 2026-10-20). The schema and every client are unchanged by that move; anything the session
process would have to grow to serve a client belongs in `Mila::AI` instead.

**As built (2026-10-24).** `mila-session` (`mila-desktop-session` until 2026-10-26, section 9.1), built beside
`mila-chat` wherever Chat builds
(`Desktop.SessionProcess.ixx`, `Desktop.ConversationStore.ixx`):

- **Settings resolve as the console's do**, from the same files and layers (`Chat.SettingsResolution`, moved
  out of `mila-chat`'s `main.cpp` so both programs share it); its flags are `--model`, `--context-length`,
  `--device`, `--system-prompt` and `--settings`, plus `--conversations <dir>` for the store of 6.3, which
  defaults to `Mila Desktop\Conversations` under `%LOCALAPPDATA%` (`$XDG_DATA_HOME/mila-desktop/conversations`
  elsewhere).
- **Activities** (3.1, 2026-10-26). A conversation is Ask & Explore, with the prompt the settings resolved and no
  tools, or Build & Code, with the `files` prompt beside the program and the file tools over the conversation's
  folder (`Desktop.FileTools.ixx`): `list_folder`, `read_file`, `find_files`, `search_text`, `write_file` and
  `edit_file`, paths relative to the folder; or Research & Analyze, with the `research` prompt and `read_web_page`
  (`Desktop.WebPageReader.ixx`): a page fetched over Mila's own HTTP transport, following at most five redirects and
  only to http and https, cut at 4 MiB, and read as text (`Desktop.HtmlText.ixx`, its own test target) in parts of
  40,000 characters. The model stays resident across the change: the session gives the agent
  the activity's prompt and tools (`ChatSession::useSystemPrompt`, over `Agent::withoutTools`), and its grant
  (`ChatSession::useGrant`, `ToolCalling.md` 3.4): none for Ask & Explore, the folder read-write for Build & Code,
  the network for Research & Analyze, both for Get Things Done. Every file tool opens through its call's grant, so
  a path, link or junction leading outside the folder is refused; the page reader and web search fetch through it,
  so they reach public addresses only. The session sets no rule for the file tools yet, so every call asks. Run on
  Gemma 4 12B Q4_0 over stdio and in the window: it listed the
  folder, read a file, fixed a bug with one exact edit, and answered a declined write as declined; in Research &
  Analyze it read a Wikipedia article and answered from it, and read a missing page's 404 as one.
- **Two threads.** The one that started the process does all session work, model loads included; a second
  reads requests. It answers `turn.stop` and `turn.tool_decision` itself, since the session's thread is inside the
  reply they are about, and refuses a `turn.send` while a turn is queued or running, so the refusal does not
  wait for it. Every other request is queued and answered in order. End of input stops a reply in flight and
  ends the process.
- **A tool call is announced when the session answers it**: `turn.tool_call` arrives after the round's
  `turn.round_end`, then `turn.tool_approval` when it asks; a tool allowed for the conversation sends
  `turn.tool_call` and `turn.tool_result` together. Call ids are the process's own; the library's `ToolCall` has
  none.
- **`thought_tokens`** is the round's reasoning text encoded again (`RoundStats::reasoning_tokens`), which can
  differ from what was sampled by a token or two; Llama, whose reply Chat parses itself, reports 0.
- **Every attachment is refused, by name.** No model this build runs takes an image or audio yet
  (`isSupportedModality`), so `session.state` lists no modality and `attachment.add` answers
  `attachment.refused`.
- **The stored conversation also keeps `turns`**, the history as the model read it (6.3), so a reopened
  conversation is prefilled exactly rather than rebuilt from what the page shows.

Gate, measured 2026-10-24 on the RTX 5060 Ti (`D:\Claude\phase3\gate.py`, scratch stores; every line on
standard output checked to be one `"v": 1` message): Llama 3.2 3B FP4 for scenarios 2, 4, 5 and 8 and a Llama
tool call; Gemma 4 26B-A4B Q4_0 for 1, 3, 6, 7 and 11 on the streamed reasoning path; Qwen 3.8 27B FP4 for a
tool call with reasoning through the template. Scenario 8 installed `gpt2-small`, the smallest published model,
into an empty store (196 progress messages, every one bytes of a total), and removed it with every file
reclaimed. It is a base model, so the load that follows an install into an empty session was refused as an
error of the install, and no published model can show that load succeeding until the Llama packages are
republished (BACKLOG, "A model installed under v0.20 stops loading").

### 6.2 The window (phase 1, against the mock session)

Starts the session process, hosts the page, and relays messages between them (section 4). Its rules
for any client that hosts the page: the page gets no file-system or process access beyond the bridge's three calls
(`send`, `onMessage`, `pathForFile`); every message from the page is checked to come from the page's
own origin; developer tools and the default context menu are off in release builds; links go to the
system browser.

**The window is Electron** (Todd, 2026-10-20: one presentation layer), in `Mila/Apps/Desktop/Window/`:
one window application for Windows, macOS and Linux. Its main process spawns the session process with
Node's child-process API and relays lines; a preload script exposes the bridge as `window.mila`, with
context isolation on and Node integration off in the page. One engine (Chromium) everywhere.

- **The page's address is `mila://chat/`**, a standard, secure scheme the window registers and serves
  from the page's own folder, nothing outside it (`Window/src/page-protocol.js`). That address is the
  origin every message from the page is checked against, and the `'self'` of section 5's policy.
- **The icon is the Mila mark**, the website's (`Web/static/mila-mark.svg`), rendered to
  `Window/icons/icon.png` (512 px) and `icon.ico`; the brand's mark is the app's.

- **Costs, accepted:** Electron ships a new major about every eight weeks and supports the latest three,
  so keeping the window's Chromium current is ours; a Node and npm toolchain beside the C++ build, for
  the window only; about a hundred megabytes of Chromium in the package.
- **On macOS** it has the platform's window pieces: vibrancy (including a `sidebar` material),
  repositioned traffic-light buttons for a frameless window, sheets for modal dialogs, rubber-band
  scrolling (Electron's `BrowserWindow` documentation).
- **To confirm from Electron's documentation before the window is written:** how a dropped `File` becomes
  a path (`webUtils.getPathForFile`).
- **Which session it starts:** the session the package carries (`resources/Session/`), or in development the
  copy staged for packaging (`Window/session/`); `MILA_SESSION` names another program, and `--scenario <n>`
  runs the mock session (section 5) instead. It answers `folder.choose` with the system's folder picker. The mock stays for development and is not packaged, so a packaged
  build has no `--scenario` (2026-10-26).
- **The package:** Electron and the packager pinned to exact versions in `package.json`, with the lock
  file committed; `node_modules/` and build output ignored by a `.gitignore` inside the folder. The
  product name is "Mila" (application id `me.toddt.mila`, section 9). electron-builder documents building
  Windows targets on Linux through Wine, or its `electronuserland/builder:wine` Docker image; the window
  has no native Node dependency, which is the case it names as needing a Windows machine. Signing is
  distribution's (section 9).

### 6.3 Conversation store

The session process owns it, so every window gets it equally: one JSON file per conversation (section
4.3 plus `id`, `title`, `model`, `activity`, `folder`, `plan`, `created`, `updated`, and `turns`, the history as the
model read it) under the
user's local application data, never in the model store. An id is used as a file name only when it is made of
the characters the store gives ids, so an id from the page cannot reach outside the folder. The title is the first user message, cut at a word under 60 characters. Opening a
conversation replaces the session's history and takes its activity and folder; its next turn prefills it whole. A
conversation whose folder no longer exists is not opened, and says so (`error`, `during: "conversation"`).

### 6.4 Attachments

The page never decodes a file. It previews it with an object URL, asks the bridge for its path, and sends
`attachment.add` with the path; the session reads and decodes it as the console's `/image` and `/audio`
do (`Gemma4Modality.md` 5.7: PNG and JPEG; WAV mixed to mono and resampled to 16 kHz). Paths, not bytes,
because the window and the session are on one machine and the session decodes anyway. Where the bridge
returns no path -- a picture dragged out of a web page, which is not a file on disk -- the page refuses
it as not saved on this computer. The attach button needs no fallback: a file the window's picker
returns always has a path, so the bridge stays at its three calls. The preview's object URL stays in
the page and is never sent. In the mock, an attachment is accepted or refused by the scenario.

## 7. The session split (phase 2, local)

`ChatSession` (`Chat.Session.ixx`) holds the model and the settings, and never draws or reads input; since
`+98` the model is a `Mila::AI::Agent`, which holds the conversation and runs the turn and its calls, and the
session words the agent's events as its own. **Events** go out through `SessionEvents` (`Chat.SessionEvents.ixx`), section
4.1 as C++ calls: `info`, `error`, `log`, a load's start and end, an install's progress (phase 3; the console
draws the line the catalogue used to write itself), a turn's start, each round's start and end
with its measurements, text by channel, a thought break and close, a tool call opening, the approval of
section 4.5 (`approveToolCall`, which the reply waits on), the answered call, and the turn's end with its
whole display. **Requests** are the session's methods, section 4.2 as calls: `respond`, `stop` (from any
thread), `newConversation`, `loadModel`, `installModel`, `removeModel`, the context and settings setters,
`seed`. A front end reads state from the session rather than being told it. Both are shaped as
`Mila::AI` will need them -- a streamed reply by channel, tool calls and their approval, rounds, the
conversation it holds -- so that v0.22.0 moves this code into Mila.AI rather than redesigning it (Todd,
2026-10-24).

- **The split follows presentation.** Routing a reply to its channels is the session's: which channel text
  belongs to is a fact about the reply, read by the template's reader in the library (`+94`; Chat's Gemma-only
  token router is deleted), and the session emits raw text. The console
  keeps the incremental rich-text formatter, the renderer, the stream validator, the spinner, the slash
  commands with their usage lines, and every table.
- **Sentences.** Those about work the session does -- a load, a reload, an install, a context that does
  not fit -- are the session's `info` and `error`; the console words its own commands. Some of the
  session's named console commands ("/model <name> loads it again"); phase 3 reworded them, so a remedy is
  said in words ("an automatic context picks the longest that fits") and each front end keeps its own
  commands, the console's in `/help`. A refusal at the shell (`RefusalAudience::Shell`) still names its
  `mila-store` command. A failed load, install or remove is the session's `error`, not its `info`, so the
  page shows a refusal where it happened.
- **The session is a library target of its own, `MilaChatSession`,** which `mila-chat` and the session
  process link. Nothing in it can import the renderer, so the build keeps the line. To be dissolved
  (2026-10-27, Todd): an agent becomes a model and a session in `Mila::AI` (`Direction.md` 4.3), and each
  application keeps only its own layer over it. The session process then reads settings of its own, which the
  Settings view writes, rather than Chat's files; its prompts live in `Mila/Apps/Desktop/`; and it offers only its
  own tools, Chat's canned weather tool gone (`Notebooks/SessionInventory.md` section 5).
- **`turn.stop` is `ChatSession::stop()`,** on a stop source made fresh for each turn; the library ends a
  generate with `GenerateStatus::ClientCancelled` on it. A tool call not yet run is not run. Ctrl+C during
  a reply calls it -- the console installs its SIGINT handler for the turn only, so Ctrl+C at the prompt
  still ends the process -- and the reply ends with "Reply stopped."
- **The console allows every call** (`approveToolCall` returns allow-once) until BACKLOG "No Chat asks
  before a tool runs" gives it the prompt line of `ToolBroker.md` section 5.

Gate, measured 2026-10-24: a piped 24-line session -- every command class, a tool call with reasoning
hidden and shown, detail all -- on Llama 3.2 3B and 3.1 8B (FP4, packaged afresh: the published packages
are refused, BACKLOG "A model installed under v0.20 stops loading"), Gemma 4 12B FP4, Gemma 4 26B-A4B Q4_0
(the streamed-reasoning path, which the 12B package does not declare) and Qwen 3.8 27B FP4; greedy, context
8192. Standard output and error are byte-for-byte equal to the build before the split once the spinner's
frames and the log timestamps are removed, and a mutant in the streamed thought break was caught. One-shot
`-p`, plain and JSON, is equal (prefill time excluded). The stream validator printed nothing. Ctrl+C
stopped a reply and the next turn answered, on Gemma, Qwen and Llama.

Two defects were found and fixed with it. `/context` ended the process on Qwen before any turn and on
Llama after any reply, because the floor rendered the history as a prompt; it now measures the history
with the next turn's user message. And a streamed word still being written when a tool call or a notice
interrupted the reply printed after the trace line; the renderer now writes it first.

## 8. Phases

1. **The page and the window, against the mock session** (cloud session; Todd, 2026-10-20), given in two
   steps, each reviewed before the next is sent. Both touch only the desktop app's folder (`Mila/Apps/Desktop/` since 2026-10-24) and, for a
   schema change, this document, which is changed first; both leave the schema in section 4 as it is
   otherwise.
   - **1a, the page** (`Ui/`). Done when every scenario in section 5 renders as section 3 describes, at
     1280x800 and at 900 px wide, in light and dark, in a browser over the in-page mock; the `node --test`
     suite passes; and the page makes no network request (checked in the browser's network log). The
     layout and scenario 1 come first, shown as screenshots, before the rest is built.
   - **1b, the window** (`Window/`, and the mock session's Node wrapper). Done when the mock session
     answers over stdio under `node --test`; the Electron window starts it and runs the same scenarios; a
     link opens in the system browser, a dropped file arrives as a path, and developer tools are off in a
     packaged build; and a Windows build exists -- an installer if the build environment allows, otherwise
     an unpacked app folder, otherwise one documented command that produces it on Windows.
2. **The session split** (local). Section 7.
3. **The session process** (local). Section 6.1. Done when a script driving it over stdio runs the
   equivalents of section 5's scenarios 1 to 8 on a real model, and the console's gate in section 7 still
   holds.
4. **The real session in the window** (local). The window starts the session process in place of the
   mock; the page and window from phase 1 are unchanged except where the real session showed the schema
   wrong, and this document is amended in the same change.
   *Run 2026-10-24* (RTX 5060 Ti; the Electron window over `mila-desktop-session` through
   `MILA_DESKTOP_SESSION`, driven over the DevTools protocol, scratch stores): Llama 3.2 3B FP4 for scenarios
   2, 3, 4, 6 (Attach disabled, with its reason) and 7 and a call that asks; Gemma 4 12B Q4_0 for 5, 1, 3 and
   11 on the streamed reasoning path, with Allow once, Deny and Allow in this conversation; 10 as 18,264
   streamed tokens in one reply (the multiplication tables to 40 x 40), with no frame over 17 ms and no long
   task; 8 as `gpt2-small` into an empty store, with progress, its refused load reported as an error of the
   install, and its removal reclaiming every file; 9 as the load's two `log` lines (the model and its memory)
   in the drawer at detail `all`. The schema was wrong in one place: a stored call (4.3). The page lacked two
   things section 3 describes, built in the same change: the log drawer, and a refused install shown in the
   Models view (it had been drawn in the transcript beneath the view, out of sight).
5. **Distribution.** Section 9.
6. **Activities** (local). Section 3: the home screen with *Ask & Explore* and its box, the sidebar's activity,
   the header's detail, the limits panel, Settings and the models view moved off the main path, and the
   first-run recommendation. Each further card arrives in the change that gives it its tools: *Build & Code*
   with the file tools, *Research & Analyze* with web reading. The selection of 3.2 is its own `Mila::AI`
   change, proposed first.

## 9. Open decisions

1. **Names -- the app's decided 2026-10-24 (Todd: "Mila is the showcase app").** The window is called
   Mila, the complete agent harness it is becoming, with the conversation as its face; a windowed program
   is not typed, so it carries no `mila-` command name, and is installed and shown in the Start menu and
   the Dock. It is its own app, `Mila/Apps/Desktop/`, beside Chat's console in `Mila/Apps/Chat/`. The session
   process's name, *decided 2026-10-26 (Todd):* `mila-session`, a program of the app's own rather
   than a mode of `mila-chat` (section 6.1). The name implies a local session; one reached over a network
   would be another program, such as `mila-network-session`.
2. **Positioning -- decided 2026-10-24 (Todd), reversing 2026-10-20.** "Mila is the brand name. [The
   desktop app] IS the showcase agentic application for end users. Mila C++23 library is the dev target
   audience." The 2026-10-20 decision -- the window built for developers deciding whether to build on
   Mila -- is withdrawn. The console stays a first-class Chat, for the developer; the app adds no
   capability of its own (agent logic and tool calling live in `Mila::AI`, `ToolCalling.md`; the app's tools
   are its own code).
3. **Which release -- decided 2026-10-24 (Todd).** Phases 2 to 4 ship in v0.21.0, with tool calling, and the
   release moves two weeks to 2026-12-04 for them. Tool calling was designed through a tool broker
   (`ToolBroker.md`); on 2026-10-25 it became `Mila::AI`'s own (`ToolCalling.md`).
4. **Distribution.** An installer, code signing (an unsigned download is warned on by SmartScreen on
   Windows; macOS needs a Developer ID and notarization) and updates. An end user installs no toolkit, so
   the app carries the CUDA runtime its session needs, under NVIDIA's redistribution terms, and states the
   GPU it needs before the download (`Untriaged.md`, 2026-10-24). *Decided 2026-10-26 (Todd):* the
   installer carries them, as Ollama's Windows package does -- no separate download and no download on first
   run. The build copies the CUDA libraries the session loads from the build machine's toolkit to beside the
   session program: at CUDA 13.4 `cublasLt64_13.dll` (about 470 MiB) and `curand64_10.dll` (about 57 MiB), the
   runtime itself linked statically (`dumpbin /dependents`, 2026-10-26). Both are in the 13.4 EULA's
   Attachment A, on its conditions: the app adds substantial function, only the app uses them, and its terms
   match NVIDIA's. *Decided 2026-10-26 (Todd):* the Visual C++ runtime ships app-local, beside the session, from
   the toolset's own redistributable folder. VS 2026's Distributable Code list covers everything under
   `VC\redist` for every edition, so the terms allow either way. Microsoft's redistributable installer needs
   an administrator and a registry version check, which a per-user installer does not have; and a machine whose
   installed runtime is older than the toolset's would not start the session. The cost is that Windows Update
   does not service an app-local runtime, so a runtime fix reaches users in Mila's next release.
   *As built 2026-10-26:* the session's `mila-session` install component (`Desktop/Session/CMakeLists.txt`)
   stages the program, its prompts and the libraries it imports -- found by reading its imports
   (`install(RUNTIME_DEPENDENCY_SET)`), so a toolkit that renames them needs no edit -- and the Visual C++
   runtime (`InstallRequiredSystemLibraries`, which needs a CMake that knows the VS 2026 toolset: Visual
   Studio's 4.3 does, 4.0 does not and is warned on) into `Window/session/`, which the package carries as
   `resources/Session/`; the window's build refuses to run without the program and the runtime.
   *Built 2026-10-27 (Todd: "I do things in vs2026 not the command line"):* the `mila-app` CMake target
   (`Desktop/CMakeLists.txt`) builds the session, stages it afresh and runs the window's build, so the installer
   and `Mila.exe` come from one target in Visual Studio. It is outside ALL, and needs `npm` found at configure.
5. **Math.** Shown as source in phase 1; a vendored math renderer later, or never.
6. **When macOS.** macOS is wanted (Todd, 2026-10-20). The page and the Electron window run there as they
   are; the session process reaches it once a Metal backend exists.
7. **Activities over models -- direction decided 2026-10-26 (Todd).** The app starts from the work, through four
   activity cards and one box, with the model, its deployment and the agent's limits chosen by Mila from what it
   has measured, and the person's overrides as caps (section 3). It replaces the layout this document had until
   then: a model picker in the header, the models view on the sidebar, and sampling on the main path. Within it:
   - *Decided with it:* the activity is the app's, and maps to needs the library ranks models by (3.2); a card
     is shown only once its work can be done (3.1); nothing guesses the activity from the text typed; there is
     no cap on parallel agents.
   - *Open:* the selection query of 3.2 (a `Mila::AI` proposal); whether a conversation may change its
     activity, at the cost of a prefill; what *Get Things Done* needs of `Mila::AI` beyond the loop it has.
   - *Considered and declined:* the model-catalogue layout of LM Studio and its peers, for the reason in
     section 1; inferring the activity from the first message.
   - *Reversed 2026-10-26 (Todd):* *Get Things Done* waiting for `Mila::AI` to run a task. It is shown now in the
     app's form (3.1): a plan the model proposes and the person approves, marked as the model works. With it, each
     activity has its own limits: Ask & Explore keeps Chat's four rounds and a reply of `max_new_tokens`, numbers no
     measurement set; the others run with none, bounded by the context and the person's Stop, until the limits of
     3.6 let the person set caps.
**Decided 2026-10-20 (Todd).** The page is HTML, so one page serves every platform. The session is its
own process and the window a client of it, so the window's language is free. The window is Electron, so
there is one presentation layer. Considered and declined:

- **WinUI 3** -- native controls, the Fluent look and no web view process. Windows-only, so nothing in it
  carries to macOS; its cross-platform routes (Uno, MAUI) are .NET, which would put the window on a .NET
  binding rather than on the session; and its XAML compiler is driven by MSBuild, where a CMake build has
  been reported missing the generated headers and Clang is unsupported (Windows App SDK issue 1156,
  open).
- **A native shell per platform, written here** -- Win32 over WebView2, `WKWebView` on macOS; no
  third-party dependency, and the system keeps each engine current. Declined for one presentation layer:
  it is one shell per platform and two engines to test the page in, and on Windows a dropped file's path
  depends on `postMessageWithAdditionalObjects`, whose documentation is not consistent on which objects
  it carries.
- **webview/webview** (an MIT C/C++ library over WebView2, WebKit and WebKitGTK) -- no third-party
  dependency for the window shell (Todd).
- **The session inside the window's process** -- the first draft of this document. It tied the window to
  C++ and to one native shell per platform, and let a fault in either side take down the other.
- **Electron with the session as a Node native addon** -- an addon is rebuilt against each Electron
  major's ABI, through C++23 modules and CUDA; the child process needs neither.
