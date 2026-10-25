# Mila Desktop

Mila, the desktop app (`Mila/Apps/Desktop/`): a sidebar of conversations and a main pane, over the same
session as Chat's console. The session runs as its own C++ process; the window is a client of it, in
whatever language suits the window. The console stays first-class on every platform.

**Status:** draft, 2026-10-20. Phase 1 (the page and the Electron window, against a mock session) is
built; phase 2 (the session split, section 7) and phase 3 (the session process, section 6.1) are built,
2026-10-24. The app's name and audience are decided (section 9, 2026-10-24). Open decisions are in section 9.

---

## 1. What this is

Chat is a session and a console front end over it: `ChatSession` (`Chat.Session.ixx`) holds the model
resolution, the deployment request, the turn loop, tool dispatch and channel routing, and the console
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

**An app adds no capability to the library** (CLAUDE.md). The window shows what the library and the
session already do. Video, which no Mila model takes, has no place in it.

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
- **Formatting is presentation.** The session emits channel-routed *raw* text; routing a token to its
  channel is the session's (`StreamingChannelRouter`). The console front end
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
  the family-specific code (Vnext: "Chat and the inference server each hold code that knows which model
  they are running") is never written.

## 3. Layout

Modelled on the Claude desktop app: a sidebar on the left, the conversation on the right.

**Sidebar**

- *New conversation* at the top (Ctrl+N).
- Conversations, newest first, grouped *Today*, *Yesterday*, *Previous 7 days*, *Older*. Each row shows
  its title; hover shows the model it was held with. Rename and delete from a row's menu; delete asks
  first.
- At the foot: *Models* (opens the models view) and *Settings*.
- Collapsible; collapsed below 900 px of window width.

**Main pane**

- *Header:* the loaded model as a picker (installed models, then *Manage models...*), and the session's
  facts as quiet text: context window (with "auto" when it was measured, not chosen) and thinking (off,
  on, or the effort level). With no model loaded, the header says so and the picker is the way out.
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
- *Composer:* a multi-line box (Enter sends, Shift+Enter is a newline), an attach button, drop anywhere on
  the pane to attach, attachments as removable chips above the box. While a reply is generating, *Send*
  becomes *Stop* (Esc). Attach is disabled with a tooltip naming why for a model without that modality.

**Models view** (a pane over the main pane): installed models with size and quantization, *Load* (with
the quantization choice for unquantized weights: none, fp8, fp4, q4_0), *Remove* (asks first);
*Available to install* from the published list, *Install* with progress. A load that is refused shows the
refusal's own text.

**Settings** (a side panel): sampling (temperature, top_k, top_p, presence_penalty, each with *model
default*), effort (1-5, when the template has levels), thinking on/off, context (a number or *auto*),
detail (off, thoughts, all), seed. Each reflects `session.state`; changing one sends `settings.set`.

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

The window relays every message unchanged in both directions, except the two it handles itself:
`link.open`, and resolving a dropped file to a path (section 6.4).

### 4.1 Session to page

| type | fields | meaning |
|---|---|---|
| `session.state` | `model` (name or null), `no_model_reason` (string or null), `context_length`, `context_automatic`, `thinking_capable`, `thinking` (`"off"`, `"on"` or an effort name), `effort` (1-5 or null), `effort_levels` (bool), `detail` (`"off"`, `"thoughts"`, `"all"`), `sampling` ({`temperature`, `top_k`, `top_p`, `presence_penalty`}, each `{value, is_model_default}`), `modalities` (subset of `["image","audio"]`), `busy` (bool), `streams` ([{`stream_id`, `parent_stream_id` (or null), `conversation_id`, `busy`}]) | The whole session state. Sent after every change, never as a delta. |
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
| `conversations.list` | `items` ([{`id`, `title`, `updated` (ISO 8601), `model`}]) | After any change to the store. |
| `conversation.loaded` | `id`, `title`, `model`, `messages` (section 4.3) | The conversation now in the main pane. |
| `attachment.accepted` | `attachment_id`, `kind`, `name`, `size_bytes` | The session has read and decoded the file. |
| `attachment.refused` | `attachment_id`, `reason` | Not found, not decodable, too large, or no such modality on this model. |
| `info` | `text` | A sentence the console would print with `printInfo`. Shown as a quiet line in the transcript. |
| `error` | `text`, `during` (`"load"`, `"turn"`, `"install"`, `"settings"`, or null) | Shown where it happened, never as a modal. |
| `log` | `level`, `text` | Only at detail `all`; shown in a collapsible log drawer. |

### 4.2 Page to session

| type | fields | meaning |
|---|---|---|
| `ready` | -- | The page has loaded; the session replies with `session.state`, `conversations.list` and `models.list`. |
| `turn.send` | `text`, `attachment_ids` | A user turn. Refused with `error` while `busy`. |
| `turn.stop` | -- | Stop the reply in flight; it ends with `finish: "cancelled"`. A call waiting for approval is not run. |
| `turn.tool_decision` | `turn_id`, `call_id`, `decision` (one of the approval's `choices`) | The person's answer to a `turn.tool_approval`. One for a call not waiting is ignored. |
| `conversation.new` | -- | Clear the session's history and start a new conversation. |
| `conversation.open` | `id` | Replace the session's history with that conversation's. |
| `conversation.rename` | `id`, `title` | |
| `conversation.delete` | `id` | |
| `models.refresh` | `online` (bool) | |
| `model.load` | `name`, `quantization` (`"none"`, `"fp8"`, `"fp4"`, `"q4_0"` or null) | Clears the history, as `/model load` does. |
| `model.install` | `name` | |
| `model.remove` | `name` | |
| `settings.set` | one of: `context` (n or `"auto"`), `thinking` (bool), `effort` (1-5), `detail`, `sampling` ({`key`, `value` or `"model"`}), `seed` | One change per message; the reply is `session.state`, or `error` with `during: "settings"`. |
| `attachment.add` | `attachment_id`, `kind`, `name`, `path` | The file on this machine; the session reads it (section 6.4). |
| `attachment.remove` | `attachment_id` | |
| `link.open` | `url` | Handled by the window, which opens it in the system browser. The page never navigates. |

### 4.3 Conversation messages

`[{ "role": "user" | "assistant" | "tool", "content": string, "thought": string | null,
"tool_calls": [{ "call_id", "name", "arguments", "result" }] | null, "attachments": [{ "kind", "name" }] | null,
"stats": { "prefill_ms", "tokens_per_second", "tokens", "rounds", "finish" } | null }]`

The same shape is the stored conversation (section 6.3). An attachment's bytes are not stored in phase 1;
a reopened conversation shows the attachment's name only.

### 4.4 Commands

Every console command has its message: `/clear` is `conversation.new`, `/model load` is `model.load`,
`/context`, `/thinking`, `/effort`, `/set`, `/verbose` and `/seed` are `settings.set`, `/stats` is the stats
line, `/model list` is `models.refresh`. The composer does not parse slash commands.

### 4.5 Approving a tool call

The session asks; the page only presents (`ToolBroker.md` section 5, which these messages carry unchanged once
the broker decides). Until the broker exists, the session's own rules decide, and a tool the person has not
allowed is asked about; afterwards the session asks when the broker's rule for that agent and tool is ask.

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

**As built (2026-10-24).** `mila-desktop-session`, built beside `mila-chat` wherever Chat builds
(`Desktop.SessionProcess.ixx`, `Desktop.ConversationStore.ixx`):

- **Settings resolve as the console's do**, from the same files and layers (`Chat.SettingsResolution`, moved
  out of `mila-chat`'s `main.cpp` so both programs share it); its flags are `--model`, `--context-length`,
  `--device`, `--system-prompt` and `--settings`, plus `--conversations <dir>` for the store of 6.3, which
  defaults to `Mila Desktop\Conversations` under `%LOCALAPPDATA%` (`$XDG_DATA_HOME/mila-desktop/conversations`
  elsewhere).
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
- **Which session it starts:** the real session process when its path is configured, and the mock
  session (section 5) otherwise. Until phase 3, every build runs the mock; afterwards the mock stays for
  development and is not packaged.
- **The package:** Electron and the packager pinned to exact versions in `package.json`, with the lock
  file committed; `node_modules/` and build output ignored by a `.gitignore` inside the folder. The
  product name is "Mila" (application id `me.toddt.mila`, section 9). electron-builder documents building
  Windows targets on Linux through Wine, or its `electronuserland/builder:wine` Docker image; the window
  has no native Node dependency, which is the case it names as needing a Windows machine. Signing is
  distribution's (section 9).

### 6.3 Conversation store

The session process owns it, so every window gets it equally: one JSON file per conversation (section
4.3 plus `id`, `title`, `model`, `created`, `updated`, and `turns`, the history as the model read it) under the
user's local application data, never in the model store. An id is used as a file name only when it is made of
the characters the store gives ids, so an id from the page cannot reach outside the folder. The title is the first user message, cut at a word under 60 characters. Opening a
conversation replaces the session's history; its next turn prefills it whole.

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

`ChatSession` (`Chat.Session.ixx`) holds the model, the conversation, the settings and the turn loop, and
never draws or reads input. **Events** go out through `SessionEvents` (`Chat.SessionEvents.ixx`), section
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

- **The split follows presentation.** Gemma's token router (`Chat.StreamingChannelRouter.ixx`) is the
  session's: which channel a token belongs to is a fact about the reply, and it emits raw text. The console
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
  process link. Nothing in it can import the renderer, so the build keeps the line.
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
5. **Distribution.** Section 9.

## 9. Open decisions

1. **Names -- the app's decided 2026-10-24 (Todd: "Mila is the showcase app").** The window is called
   Mila, the complete agent harness it is becoming, with the conversation as its face; a windowed program
   is not typed, so it carries no `mila-` command name, and is installed and shown in the Start menu and
   the Dock. It is its own app, `Mila/Apps/Desktop/`, beside Chat's console in `Mila/Apps/Chat/`. *Open:*
   the session process's name. It is built as `mila-desktop-session`, a program of the app's own rather than
   a mode of `mila-chat` (section 6.1); no one types it, so the name is the app's to keep or change.
2. **Positioning -- decided 2026-10-24 (Todd), reversing 2026-10-20.** "Mila is the brand name. [The
   desktop app] IS the showcase agentic application for end users. Mila C++23 library is the dev target
   audience." The 2026-10-20 decision -- the window built for developers deciding whether to build on
   Mila -- is withdrawn. The console stays a first-class Chat, for the developer; the app adds no
   capability of its own (agent logic lives in Mila.AI and the tool broker).
3. **Which release -- decided 2026-10-24 (Todd).** Phases 2 to 4 ship in v0.21.0, with tool calling through
   the tool broker (`ToolBroker.md`), and the release moves two weeks to 2026-12-04 for them.
4. **Distribution.** An installer, code signing (an unsigned download is warned on by SmartScreen on
   Windows; macOS needs a Developer ID and notarization) and updates. An end user installs no toolkit, so
   the app carries the CUDA runtime its session needs, under NVIDIA's redistribution terms, and states the
   GPU it needs before the download (`Untriaged.md`, 2026-10-24).
5. **Math.** Shown as source in phase 1; a vendored math renderer later, or never.
6. **When macOS.** macOS is wanted (Todd, 2026-10-20). The page and the Electron window run there as they
   are; the session process reaches it once a Metal backend exists.
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
