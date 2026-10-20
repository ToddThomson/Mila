# Chat Desktop

A windowed front end for Chat on Windows: a sidebar of conversations and a main pane, over the same
session as the console. Linux keeps the console.

**Status:** draft, 2026-10-20. Phase 1 (the window's contents, against a mock) is written to be built by
a cloud session from this document alone. Open decisions are in section 9.

---

## 1. What this is

Chat today is one console program: `Chat.ixx` holds the session (model resolution, the deployment
request, the turn loop, tool dispatch, channel routing) and writes to the terminal itself, through a
concrete `ConsoleRenderer` and `std::getline( std::cin )`. A window is a second front end over that same
session. It exists because the things a person now does with a local model are window-shaped:

- **Media.** An image or an audio clip attached by drag-and-drop, shown in the transcript, and refused by
  name for a model that has no such modality (`Gemma4Modality.md` 5.7). A terminal can take a path; it
  cannot show the picture back.
- **Conversations.** A sidebar of past conversations, reopened where they were left. Chat has no
  conversation store today: every session starts empty.
- **A showcase for developers.** Chat, like the inference server, is an application that shows what can
  be built on Mila, and the window is where that shows best. Its audience is the developer deciding
  whether to build on Mila, not someone choosing a desktop chat app (decided 2026-10-20, section 9).

**An app adds no capability to the library** (CLAUDE.md). The window shows what the library and the
session already do. Video, which no Mila model takes, has no place in it.

## 2. Architecture

```
  +---------------------------+          +-------------------------------+
  |  Console front end        |          |  Desktop front end (Windows)  |
  |  ConsoleRenderer, stdin   |          |  Win32 window + WebView2      |
  +-------------+-------------+          |  host  <-- JSON -->  Ui/      |
                |                        +---------------+---------------+
                |   events out / requests in (section 4) |
                +-------------------+--------------------+
                                    |
                     +--------------+---------------+
                     |  Chat session (Chat.ixx,     |
                     |  less its console I/O)       |
                     +--------------+---------------+
                                    |
                                  Mila
```

- **The line is presentation versus session.** It is the line v0.22.0 draws when Chat is rebuilt on
  `Mila::AI` ("Chat keeps terminal rendering and the approval gate", `Vnext.md`): what is beneath it moves
  then, and neither front end does.
- **Formatting is presentation.** The session emits channel-routed *raw* text. The console front end
  keeps `IncrementalRichFormatter` and `RichText` (markdown and LaTeX to terminal Unicode, word wrap); the
  window renders markdown itself, in the page. Each front end keeps its own streamed-equals-buffered
  check.
- **The page is HTML, CSS and JavaScript in WebView2.** The host is a thin Win32 program: it creates the
  window and the WebView2 control, serves `Ui/` from disk, and moves JSON both ways. All layout,
  rendering and interaction live in `Ui/`.
- **The session runs in the window's process, on the library directly; not through the inference
  server.** The server's wire is the OpenAI and Anthropic APIs, stateless per request, and has no model
  install, load or remove, no progress, no context rebuild and no session settings; giving it those would
  make it an app back end, which it is not. It would also put a Python runtime and a local port in an
  end-user install. And the session is C++, which is what carries to another platform (section 9).
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
    occurrence: the tool's name and summary, the result expandable.
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
creates the thing (the host for turns and conversations, the page for attachments).

### 4.1 Host to page

| type | fields | meaning |
|---|---|---|
| `session.state` | `model` (name or null), `no_model_reason` (string or null), `context_length`, `context_automatic`, `thinking_capable`, `thinking` (`"off"`, `"on"` or an effort name), `effort` (1-5 or null), `effort_levels` (bool), `detail` (`"off"`, `"thoughts"`, `"all"`), `sampling` ({`temperature`, `top_k`, `top_p`, `presence_penalty`}, each `{value, is_model_default}`), `modalities` (subset of `["image","audio"]`), `busy` (bool) | The whole session state. Sent after every change, never as a delta. |
| `models.list` | `installed` ([{`name`, `size_bytes`, `quantization`, `family`}]), `published` ([{`name`, `size_bytes`, `quantization`, `installed`}] or null when not fetched) | Reply to `models.refresh`, and after install or remove. |
| `progress` | `task` (`"load"`, `"install"`, `"rebuild"`), `label`, `done`, `total` (both null when unknown) | A long task's progress; a final one has `done == total`. |
| `turn.begin` | `turn_id`, `conversation_id` | A reply starts. |
| `turn.text` | `turn_id`, `round`, `channel` (`"thought"` or `"answer"`), `text` | Raw text to append to that channel. Gemma streams many; Llama and Qwen send one per round when it is complete. The page must render both. |
| `turn.thought_break` | `turn_id`, `round` | The reasoning channel resumed after a tool call; start a new thought block. |
| `turn.tool_call` | `turn_id`, `round`, `call_id`, `name`, `summary`, `arguments` (JSON string) | The model called a tool. |
| `turn.tool_result` | `turn_id`, `call_id`, `result`, `is_error` | The tool's result, fed back to the model. |
| `turn.round_end` | `turn_id`, `round`, `prefill_ms`, `tokens_per_second`, `tokens`, `finish` | One generate call ended. |
| `turn.end` | `turn_id`, `finish` (`"stop"`, `"length"`, `"context"`, `"cancelled"`, `"error"`), `answer` (the parsed final answer) | The reply ended. `answer` is what the conversation keeps. |
| `conversations.list` | `items` ([{`id`, `title`, `updated` (ISO 8601), `model`}]) | After any change to the store. |
| `conversation.loaded` | `id`, `title`, `model`, `messages` (section 4.3) | The conversation now in the main pane. |
| `attachment.accepted` | `attachment_id`, `kind`, `name`, `size_bytes` | The host has read and decoded the file. |
| `attachment.refused` | `attachment_id`, `reason` | Not decodable, too large, or no such modality on this model. |
| `info` | `text` | A sentence the console would print with `printInfo`. Shown as a quiet line in the transcript. |
| `error` | `text`, `during` (`"load"`, `"turn"`, `"install"`, `"settings"`, or null) | Shown where it happened, never as a modal. |
| `log` | `level`, `text` | Only at detail `all`; shown in a collapsible log drawer. |

### 4.2 Page to host

| type | fields | meaning |
|---|---|---|
| `ready` | -- | The page has loaded; the host replies with `session.state`, `conversations.list` and `models.list`. |
| `turn.send` | `text`, `attachment_ids` | A user turn. Refused with `error` while `busy`. |
| `turn.stop` | -- | Stop the reply in flight; it ends with `finish: "cancelled"`. |
| `conversation.new` | -- | Clear the session's history and start a new conversation. |
| `conversation.open` | `id` | Replace the session's history with that conversation's. |
| `conversation.rename` | `id`, `title` | |
| `conversation.delete` | `id` | |
| `models.refresh` | `online` (bool) | |
| `model.load` | `name`, `quantization` (`"none"`, `"fp8"`, `"fp4"`, `"q4_0"` or null) | Clears the history, as `/model load` does. |
| `model.install` | `name` | |
| `model.remove` | `name` | |
| `settings.set` | one of: `context` (n or `"auto"`), `thinking` (bool), `effort` (1-5), `detail`, `sampling` ({`key`, `value` or `"model"`}), `seed` | One change per message; the reply is `session.state`, or `error` with `during: "settings"`. |
| `attachment.add` | `attachment_id`, `kind`, `name`, plus the file (section 6.3) | |
| `attachment.remove` | `attachment_id` | |
| `link.open` | `url` | The host opens it in the system browser. The page never navigates. |

### 4.3 Conversation messages

`[{ "role": "user" | "assistant" | "tool", "content": string, "thought": string | null,
"tool_calls": [{ "call_id", "name", "arguments", "result" }] | null, "attachments": [{ "kind", "name" }] | null,
"stats": { "prefill_ms", "tokens_per_second", "tokens", "rounds", "finish" } | null }]`

The same shape is the stored conversation (section 6.2). An attachment's bytes are not stored in phase 1;
a reopened conversation shows the attachment's name only.

### 4.4 Commands

Every console command has its message: `/clear` is `conversation.new`, `/model load` is `model.load`,
`/context`, `/thinking`, `/effort`, `/set`, `/verbose` and `/seed` are `settings.set`, `/stats` is the stats
line, `/model list` is `models.refresh`. The composer does not parse slash commands.

## 5. The page (`Mila/Apps/Chat/Desktop/Ui/`)

- **Plain HTML, CSS and JavaScript modules. No build step, no framework, no network.** The page loads
  only its own files: a `Content-Security-Policy` of `default-src 'self'` in `index.html`, no CDN, no web
  fonts fetched. It ships as files beside the executable.
- **A third-party library is vendored**, one file each, with its licence read at its source and a
  `NOTICE.md` entry, as Chat's image decoder is (`Gemma4Modality.md` 8, decision 3). A markdown renderer is the only
  expected one; a small hand-written renderer for the subset in section 3 is equally acceptable.
- **Model text is untrusted.** It never reaches `innerHTML` unescaped: raw HTML in a reply renders as
  text, links are `link.open`, images in markdown are not fetched.
- **One transport interface** (`transport.js`): `send( message )` and `onMessage( handler )`. Two
  implementations: WebView2 (`window.chrome.webview.postMessage` and its `message` event, data in
  `event.data`) and a mock, chosen when `window.chrome?.webview` is absent. A macOS `WKWebView` host adds a
  third later (section 9), so nothing outside `transport.js` names WebView2 or relies on Chromium-only
  behaviour: the page must also run in Safari.
- **State is reduced from messages**, in a module with no DOM access (`state.js`: `reduce( state,
  message ) -> state`), so the page's behaviour is testable without a browser.
- **The mock** (`mock/`) plays scripted scenarios over the schema with realistic timing, and answers
  requests as the host would. Scenarios, each selectable from a dev-only menu (`?scenario=` in the URL):
  1. Gemma reply streamed: thinking, a `get_weather` tool call and result, more thinking, an answer with
     a table and a code block.
  2. Llama reply buffered: one `turn.text` per round, at the end.
  3. A reply stopped by the user mid-stream.
  4. A reply ending on `length`, and one on `context`.
  5. No model loaded, then a load with progress, then a load refused with the refusal's text.
  6. An image attached to a model with the modality, and refused on one without.
  7. Five conversations across the date groups; open, rename, delete.
  8. Install with progress from the published list; remove.
  9. Detail `all`: log lines in the drawer.
  10. A reply of 20,000 tokens, to see that appending stays smooth.
- **Tests:** `node --test` over `state.js`, the markdown renderer and the mock's scenario scripts. No
  test dependency beyond Node itself.

## 6. The host (phase 3, local)

### 6.1 Window and control

A Win32 program in C++23, built only `if(WIN32)` in `Mila/Apps/Chat/CMakeLists.txt`, linking the Chat
session library (phase 2). It hosts one WebView2 control filling the window, serves `Ui/` under a virtual
host name mapped to the folder beside the executable, validates every message's source before acting on
it, turns off the default context menu and developer tools in release builds, and runs the session on a
worker thread: generation never blocks the window, and every message to the page is posted from the UI
thread. Host to page is `PostWebMessageAsJson`; page to host is `add_WebMessageReceived` with
`TryGetWebMessageAsString`.

To confirm against Microsoft's documentation before the host is written: the virtual host mapping
call (`SetVirtualHostNameToFolderMapping`), how the WebView2 SDK is taken into a CMake and Ninja build,
and the Evergreen runtime's presence on Windows 10 machines (it is expected on Windows 11).

### 6.2 Conversation store

The host owns it: one JSON file per conversation (section 4.3 plus `id`, `title`, `model`, `created`,
`updated`) under the user's local application data, never in the model store. The title is the first
user message, cut at a word under 60 characters. Opening a conversation replaces the session's history;
its next turn prefills it whole.

### 6.3 Attachments

The page never decodes a file; it previews it with an object URL and passes it to the host, which
decodes it as the console's `/image` and `/audio` do (`Gemma4Modality.md` 5.7: PNG and JPEG; WAV mixed to
mono and resampled to 16 kHz). The intended route for a dropped file is
`chrome.webview.postMessageWithAdditionalObjects`, which gives the host the file's path; Microsoft's
documentation is not consistent on which object types that call carries, so the `ScenarioDragDrop`
sample is read first. The fallback is the host's own file dialog, which the attach button can use either
way. In the mock, an attachment is accepted or refused by the scenario.

## 7. The session split (phase 2, local)

`Chat.ixx` emits section 4's events through an interface and takes its requests from one, instead of
writing to `ConsoleRenderer` and reading `std::cin`; one type per file, as every touched module is
brought under the rule. The console front end implements both over today's renderer and stdin loop, so
its output does not change. `turn.stop` is wired to `stop_src_.request_stop()`; the library already ends
a generate with `GenerateStatus::ClientCancelled` on it, and nothing in Chat calls it today, so Ctrl+C
in the console stops the reply instead of ending the process. `StreamingResponseDisplay` takes the
interface rather than `ConsoleRenderer&`.

Gate: the console's transcript is byte-for-byte unchanged on the four validated models, `ChatRichTextTests`
and the stream validator pass, and Ctrl+C stops a reply.

## 8. Phases

1. **The page, against the mock** (cloud session). Done when every scenario in section 5 renders as
   section 3 describes, at 1280x800 and at 900 px wide, in light and dark; the `node --test` suite passes;
   the page makes no network request (checked in the browser's network log); and the schema in section 4
   is unchanged, or changed here first. Touches only `Mila/Apps/Chat/Desktop/Ui/` and, for a schema
   change, this document.
2. **The session split** (local). Section 7.
3. **The host** (local). Section 6, with the page from phase 1 unchanged except where the real host
   showed the schema wrong, and this document amended in the same change.
4. **Distribution.** Section 9.

## 9. Open decisions

1. **The executable's name.** Commands are lowercase `mila-` names; a windowed program is not typed, but
   is installed and shown in the Start menu.
2. **Positioning -- decided 2026-10-20 (Todd).** The window does not make Chat an end-user product. Chat
   and the inference server are applications that show developers, Mila's audience (`Direction.md` 1),
   what can be built on it; the window is built to be beautiful for that reason. The console stays a
   first-class Chat.
3. **Which release.** On `dev` before v0.21.0 is tagged, phases 2 and 3 are scope added to a release dated
   2026-11-20.
4. **Distribution.** An installer, code signing (an unsigned download is warned on by SmartScreen) and
   updates.
5. **Math.** Shown as source in phase 1; a vendored math renderer later, or never.
6. **When macOS.** macOS is wanted (Todd, 2026-10-20). The page and the session carry; the host does not.
   Once a Metal backend exists (`Future.md`, "Compute backends beyond CUDA"), a host over the system's
   `WKWebView` serves the same `Ui/`, with a third `transport.js` implementation for its message handlers
   (API names to be confirmed from Apple's documentation). So nothing in `Ui/` may assume WebView2 outside
   `transport.js`. A Linux host over WebKitGTK would follow the same pattern and is not planned.

**Decided 2026-10-20 (Todd): WebView2, because macOS is wanted.** Considered: WinUI 3, which gives native
controls, the Fluent look and no web view process. It is Windows-only, so nothing in it carries to macOS;
its cross-platform routes (Uno, MAUI) are .NET, which would put the window on a .NET binding rather than
on the C++ session; and its XAML compiler is driven by MSBuild, where a CMake build has been reported
missing the generated headers and Clang is unsupported (Windows App SDK issue 1156, open). Not measured
here: C++/WinRT under C++23 modules.
