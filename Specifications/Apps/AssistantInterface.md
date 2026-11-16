# Mila Assistant Interface

**Status:** design, round 2 (2026-11-16). Round 1, the same morning, set out projects, processes and creations around
a home screen and a sidebar of conversations; its phase 1, the home screen read from today's records, is built. Round 2
is the layout Todd reached through a working prototype the same afternoon, iterated with him on the Mac: a sidebar,
a workspace overview, projects that open on their summary, chats that need no project, and Mila as a bar at the foot
of one pane. Todd called it "a good first approximation". Every answer here is Claude's, provisional until Todd settles
it; of four that went against earlier decisions, two are settled (one workspace; general purpose, no activity cards)
and two are open in section 11. Which release each phase lands in is Todd's.

The prototype is beside this document, `AssistantInterface.prototype.html`: one page, example work, and replies
canned by keyword, since nothing runs a model behind it. It is the reference for how a screen looks and moves; this
document decides what is built. Where the two disagree, this document is right and the prototype is out of date.

How a person works with Mila Assistant: what they see when it opens, where their work lives, how they move through it
by talking to Mila, and how what Mila did to make something stays in view. It replaces the sidebar, home screen and
activities and their cards of `../MilaDesktop.md` 3.1, 3.3 and 3.4 as its phases are built, and keeps the rest of that document:
the conversation's transcript (3.5), the limits (3.6), Settings (3.7), the session and its schema (4, 6).

---

## 1. The idea

**Mila is not a project management application with AI capabilities. It is an environment for interacting with
intelligence, where structure emerges from the interaction** (Todd, 2026-11-16, decided). The central principle: the
person brings intent, and intelligence helps develop that intent into understanding, structure, processes and
creations. Nobody has to know whether something is a project, a research task, an experiment or a document before
starting. A person may begin with something as vague as *I've been thinking about how autonomous agents manage their
own reasoning loops*; Mila explores with them, and as the discussion develops it may recognize a research subject, the
questions around it, a need to look at existing approaches, an experiment worth running, and the findings and
creations that come out of it. The structure emerges because the work demands it.

**Intelligence-centric, with projects as the structure** (Todd, 2026-11-16, decided). The person interacts with Mila
naturally, without deciding where something belongs; Mila helps recognize emerging objectives, establish relationships,
create projects and organize the resulting work. Nobody manages their workspace before they can use the intelligence:
it evolves from their interactions. This is also the original intent of Mila::AI, now `Mila::Agents`.

**One workspace for each person** (Todd, 2026-11-16, decided "for now"; section 11, question C).

**General purpose: no activity cards** (Todd, 2026-11-16, decided): "the Mila assistant no longer has cards. It is
general purpose and works with the user to just do work. The idea is that projects take on structure as they evolve."
Every chat starts with every tool, and what a tool may reach is asked for the first time the work needs it (section
5.5; section 11, question B).

**Not model-centric** (Todd, 2026-11-16, decided). The person does not much care about the model. It is there and
observable, never the subject of a screen: the workspace and its projects are what the person cares about. This
carries `../MilaDesktop.md` 1's "the person chooses the work; Mila chooses the intelligence" into the layout: the model
is a quiet fact at the foot of the sidebar.

**The workspace is the top level; projects are its containers; a chat needs neither** (Todd, 2026-11-16). A chat is the
unit of work, and a project is what several chats can grow into. A person who only ever chats has a workspace of chats,
and nothing in the window asks them to make a project.

Five rules follow, and every section below is held to them:

1. **Start anywhere.** The Mila bar at the foot of every screen takes anything, and a chat begins in no project.
2. **Structure grows out of the work.** A chat joins a project when it has substance, by the person's move or Mila's.
3. **The person decides their structure.** Mila suggests a project, a move or a link; how far it may act without a
   yes is open (section 11, question A). Every change is in the workspace's activity, and can be undone.
4. **What was made keeps how it was made.** A file Mila wrote answers what it was for, which chat and plan step made
   it, which model, what Mila read first, every write since, and whether it changed on disk after.
5. **The page shows records; it holds none.** Every view is drawn from what the session stores, so another window, or
   Chat, could show the same work.

## 2. Where it stands today

- **The sidebar is one list** (`../MilaDesktop.md` 3.3): every conversation, newest first, grouped by date, each row
  its title and its activity.
- **The home screen** (3.4) is one line, the activity cards, and since phase 1 (`0.21.0-dev+147`, uncommitted when
  written) the work so far beneath them: *Needs you*, *Plans*, *Recently made*, *Recent work*, *Ideas*, read from each
  conversation's summary in `conversations.list` (`Desktop.ConversationSummary`; `homeSections` in `state.js`).
- **A conversation carries its activity and, for Build & Code and Get Things Done, a folder** chosen when it starts,
  in the window's picker (`folder.choose`). The activity decides its tools and their grant (`../ToolCalling.md` 3.4).
- **The store** is one JSON file per conversation in `%LOCALAPPDATA%\Mila\Conversations` on Windows
  (`Desktop.ConversationStore.ixx`): its messages, each tool call with its arguments, decision and result, the history
  the model read, and its activity, folder and plan.
- **Get Things Done has a plan** (`Desktop.TaskPlan.ixx`): `propose_plan`, approved by the person, and `update_step`.
- **Nothing outlives a conversation** but the conversation: no instructions, files or findings shared between them.
- **The library's task** (`../Agents/AgentTasks.md`, agreed 2026-11-03, not built) gives an agent a task carried
  across responses, a plan, a versioned trace of every step, and a task folder holding them.
- **The session holds one conversation's state at a time** (one stream, `../MilaDesktop.md` 4, *Streams*).

## 3. The concepts

| Concept | The person's word | What it is | Whose |
|---|---|---|---|
| Workspace | *Workspace* | Everything the person has with Mila: projects, chats, activity | The app's store |
| Project | *Project* | A lasting body of work: name, purpose, instructions, folders, its chats and all below | The app's store |
| Chat | *Chat* | The person and Mila talking. In one project, or in none | The app's store, as conversations are today |
| Plan and run | *Plan*, *run* | The steps of a piece of work, and each time they were worked through | The app's store, then the library's task (section 7) |
| Made | *Made* | A file Mila wrote, or a reply the person kept, with where it came from | The app's store, from the calls' records |
| Finding | *Finding* | Something learned, with the sources it rests on | The app's store |
| Event | *Activity* | One thing that happened: never edited | The app's store |
| Understanding | *What Mila understands* | A project's summary, its open questions and its next step, in Mila's words | The app's store, written by Mila (section 6) |

Three words, chosen against the repository's vocabulary:

- **Workspace, in the window.** It is Todd's word for the top level. In the library *workspace* names scratch
  buffers, and `../Agents/AgentTasks.md` decision 5 kept it out of the task's names for that reason; that rule is about
  identifiers in `Lib/Src`. The app's own code says `Workspace` only where it means this, and never beside the
  library's.
- **Chat, in the window; conversation in the code and the schema.** The messages stay `conversation.*`.
- **Made, not artifact.** *Artifact* keeps its build sense (CLAUDE.md). The spec's own noun is *creation*.

**What is the library's and what is the app's.** Grouping work, keeping projects, creations and findings, and showing
them is the app's: retrieval stores are an application's concern (`../Direction.md` 6.4), and an app adds no capability
to the library (CLAUDE.md). Giving a chat its project's instructions, understanding and findings is the app writing a
system prompt (`Session::withSystemPrompt`). Kept as the model's own state, memory would be `Direction.md` 6.4's, and the
library's. A run's record is the library's once a task exists (section 7).

## 4. The window

```
+------------------------+-------------------------------------------------------------+
| [M] Your Workspace  ^v | [::] Your Workspace / Projects / Calculator app   (example) |
| (/) Search       Cmd K +-------------------------------------------------------------+
|------------------------|                                                             |
| [::] Workspace         |   the workspace overview, all projects, a project,          |
| [ ] Projects         5 |   all chats, a chat, or activity                            |
| ( ) Chats           13 |                                                             |
| /\/ Activity           |                                                             |
| PROJECTS             5 |                                                             |
|  * Calculator app      |                                                             |
|  * Local LLMs on 16 GB |          +-------------------------------------------+      |
|  + New project         |          | the conversation, risen while you talk    |      |
| RECENT CHATS           |          +-------------------------------------------+      |
|  o Why does a KV cache |          | [M] Ask Mila: open a project, ...      (^) |      |
| Settings               |          +-------------------------------------------+      |
| . Gemma 4 12B    Model |                                                             |
| (TT) Name, owner    ^v |                                                             |
+------------------------+-------------------------------------------------------------+
```

One pane beside a sidebar. The pane is never split; the conversation with Mila rises over its lower part only while
the person talks (section 5).

### 4.1 Sidebar

Always the same places, with where the person is marked among them.

- **The workspace**, at the top: Mila's mark, *Your Workspace* and *Mila Assistant*. A click opens the overview. There
  is one workspace, so nothing switches it (question C).
- **Search** with its shortcut (Command+K, Ctrl+K): jump to any project, chat or file by name (section 4.11).
- **The four places**, each with its count where it has one: *Workspace* (the overview, 4.3), *Projects* (4.4),
  *Chats* (4.6), *Activity* (4.8). The one shown is marked.
- **Projects**: the five most recently active, each a dot of its state (amber waiting on you, teal active, grey
  quiet) and its name, the open one marked; then *New project*, which asks Mila to start one. Left out when the
  workspace has none.
- **Recent chats**: the four most recent, in any project or none; the open one marked.
- **At the foot**: *Settings* (Command+, or Ctrl+,), the window of `../MilaDesktop.md` 3.7; the model, its name and
  *Model*, which opens what Mila chose (the model, chosen for this computer, the context, thinking, the share of the
  open chat's context); and the person: initials, name, *Workspace owner*, and a menu of Settings, Mila's licence,
  About Mila Assistant and Quit.

### 4.2 The bar

Where the person is, as a path they can step back along: *Your Workspace / Projects / Calculator app*, or *Your
Workspace / Kyoto in November / Plan a three-day trip to Kyoto* for a chat. Each part but the last opens its place.
Nothing else: no model, no switch of view, no search.

### 4.3 The workspace overview

What the window opens on. Top to bottom, a section left out when it has nothing:

1. **The date and *Your Workspace*.**
2. **At a glance**: a row of figures, each a number, a word and a line, each opening what it counts. With projects:
   *Projects* (*2 with work under way*), *Waiting on you* (the projects, by name; amber when not zero), *Chats* (*5 not
   in a project*). With none: *Chats* (*Newest: ...*) and *Waiting on you*.
3. **Needs you, and what Mila noticed**: a plan stopped for the person (*Stopped at step 3 of 5: ...*, *Answer and
   continue*), a reply cut short (*Ask Mila to finish*), a file changed on disk after Mila wrote it (*Where it came
   from*), and Mila's suggestion of a project (section 6). Each names its project.
4. **Recent projects**: the three most recent as cards, each its name, purpose, state, a strip of its last fourteen
   days of activity, its latest event and its counts (*2 chats · 3 made*); *All N projects* beside the heading.
5. **Chats**: the five most recent not in a project, each its title, the start of Mila's last reply, and when; a
   dashed mark on those Mila sees as belonging together. *All N chats* beside the heading.

Not on it, by Todd's trim the same afternoon: the files made (*Recently made*), and any buttons of suggestions beside
the Mila bar. Phase 1's panels give way to this at phase 2: their data stays, their layout goes.

### 4.4 All projects

Every project: a search, a filter (*All*, *Waiting on you*, *Active*, *Quiet*), a sort (most recent, by name), and one
row each: name and purpose, state, latest event, counts. Choosing a row opens the project.

### 4.5 A project

A project opens on its summary, so the person knows where it stands before anything else:

- **The head**: *Project*, its name, its purpose, its state, its folders, and its latest event.
- **At a glance**: *Chats*, *Made*, *Plans*, *Findings*, each with a line and opening its section.
- **The summary**: *What Mila understands* (a paragraph, then *Open questions*) beside *Next* (one sentence and, where
  there is one, the one action that moves it: *Answer and continue*, *Ask Mila to finish*).
- **Then its contents**: *Chats*, *Made*, *Plans*, *Findings*, *History*. On a wide window a column beside them lists
  *In this project* with counts; the sidebar already lists the other projects.

### 4.6 All chats

Every chat, newest first: a search, a filter (*All*, *Not in a project*, *In a project*) when the workspace has
projects, and one row each with its project, or *No project*, the start of Mila's last reply, and when.

### 4.7 A chat

A chat opened from anywhere has the pane to itself: *Chat*, its title, where it is (*In Kyoto in November*, or *Not in a
project*) and when, then the conversation in a reading column, the transcript of `../MilaDesktop.md` 3.5. The Mila bar
below becomes its reply box (*Reply to Mila*). A chat begun in the risen conversation (section 5) opens whole from that
conversation's expand control.

### 4.8 Activity

Everything that happened across the workspace, by day, newest first: the person's decisions, what Mila made, plans
approved and stopped, findings, chats started. A filter by kind. Each entry says who (Mila or *You*) and where, and
opens what it names.

### 4.9 Drawers

Over the pane, from the right, closed by Esc or outside it:

- **Made**: the file's name and path, its contents where the page can show them, *Changed on disk after Mila last wrote
  it* when it did, and *Where it came from*: for which project, in which chat, during which plan step, what Mila read
  first, by Mila with which model, and every write.
- **A plan**: its goal, its steps and their state, its runs and how each ended, *Continue this run* and *Run again*.
- **Settings**, until the app's Settings window takes the place (`../MilaDesktop.md` 3.7).

### 4.10 Narrow windows

Below 900 px the sidebar leaves the window and slides over it from a menu button in the bar, closing once the person
has gone somewhere from it. The overview's figures go two to a row, a project's column of contents goes, and the
conversation may rise higher. Nothing scrolls sideways.

### 4.11 Search

Command+K or Ctrl+K, or *Search* in the sidebar: one box over the pane, projects first, then chats and files, each with
where it is; arrow keys and Enter.

## 5. Mila: the bar, and moving through the workspace by talking

### 5.1 The bar and the conversation

**The Mila bar** sits at the foot of every screen, centred in the pane: Mila's mark and one line saying what can be
asked here (*Ask Mila: open a project, see what changed, or start something new*; in a project *Ask Mila about
Calculator app, or go anywhere in your workspace*; with no projects *Ask Mila anything, or open one of your chats*). The
pane fades out beneath it, so it never sits over a card.

**The conversation rises** above the bar when the person sends something, no higher than about half the window, so the
place Mila takes them stays in view. Its head says what it is about (*Talking about your workspace*, *In Calculator
app*, *Not in a project*), with *Open as a whole chat*, *New conversation*, and lower. Esc lowers it.

**No buttons of suggestions.** Nothing is offered beside the bar or under Mila's replies; the person types (Todd's
trim, 2026-11-16).

### 5.2 Going somewhere by asking

The person can move through the workspace by asking: *show all my projects*, *which projects are waiting on me?*, *open
the calculator project*, *what's next here?*, *what did Mila make here?*, *show all my chats*, *back to my workspace*.
Mila answers in a sentence or two, takes the pane there, and leaves a line in the conversation saying where it went
(*Opened Calculator app*).

**In the app this is the model's, through tools of the app's own** that change nothing outside Mila and so run
without asking: `open_place` (`place`: *workspace*, *projects*, *chats*, *activity*; or a project or chat by name, from
the names the system prompt lists) and `find` (words; returns the projects, chats and files that match). The
prototype's keyword routing only stands in for them. Whether models call them well, and do not call them when the
person is only talking, is measured on Gemma 4 12B Q4_0 before they are on by default, and on every model in a final
run (`ToolCalling.md`'s working model; Llama's over-calling, 2026-10-25, may make its default differ).

**A request to go somewhere is not part of a chat.** Asked from inside one, it begins a conversation of its own about
the workspace, and the chat keeps only its own turns.

### 5.3 Starting something new

Something new typed into the bar begins a chat in no project, shown at once in *Chats* and marked arriving. Mila
replies, and places it as the conversation shows what it is:

- **Started inside a project**, it is that project's.
- **Close to chats Mila already sees as belonging together**, Mila says so and suggests gathering them (section 6).
- **Close to an existing project**, Mila puts it there and says so, with *keep it separate* to undo it; or only
  suggests the move. Which is question A.
- **Otherwise** it stays a chat of its own: *If it grows into one, Mila will suggest it.*

### 5.4 Answered in words

An offer Mila makes in the conversation is answered by replying: *yes* takes it, *no* or *not now* declines it,
*keep it separate* undoes a placement. The next thing the person says closes the offer either way. On the overview,
Mila's suggestion keeps its card with *Make the project* and *Not now* (question D).

### 5.5 A general-purpose chat

Decided by Todd, 2026-11-16 (question B): no activity cards. A chat is not started as a kind of work; it is work, and
what it needs is asked for when it first needs it. This replaces the activities of `../MilaDesktop.md` 3.1, which
decided a conversation's tools, their grant, its system prompt and its limits when it began.

**Every chat is offered every tool the app has**: the file tools (`list_folder`, `read_file`, `write_file`,
`edit_file`), web reading, web search when the person has given a search key, the plan (`propose_plan`, `update_step`),
running a command when that setting is on, and the app's own tools of sections 5.2 and 6. A tool whose work cannot be
done here is left out rather than offered and refused: search without a key, commands with the setting off or no
container (`../ContainedPrograms.md` 5).

**What a tool may reach is asked for when the work first needs it.** The grant is the chat's (`Session::withGrant`,
`../ToolCalling.md` 3.3), set by the app and only ever narrowed for a call, never widened by one. So a chat begins
with:

- **No folder.** When the work needs files, Mila calls the app's own `request_folder` (`why`: one line), which is
  always asked: the person picks a folder in the window's picker, or one of the project's, and the chat's grant gains
  it, read-write, for the rest of the chat. Declined, Mila goes on without files. In a project whose chats share a
  folder, the grant holds it from the start (section 10, *standing grant*).
- **The network, for reading public pages and searching,** as Research & Analyze and Get Things Done have it today.
  Reading a page runs by the rules those activities have; a search is asked every time, since its words leave the
  machine (`../MilaDesktop.md` 3.7).
- **The rules of today's activities, merged**: what any activity allowed without asking is allowed; what any asked
  about is asked; a command is asked every time, and no rule allows it (`../ContainedPrograms.md` 5).

A grant widened in a chat is that chat's: the next chat begins without it unless both are in a project that holds it.
Being in a project never widens what a tool may reach beyond the project's folders.

**One system prompt** in place of the activities' seven (`Apps/Desktop/Prompts`): what Mila is, how it uses its tools,
when it asks for a folder, when it proposes a plan or a project, and how it moves the person through the workspace. The
limits of a reply are the person's caps (`../MilaDesktop.md` 3.6), one set for every chat.

**The cost, measured before it ships:** every tool's definition is in every chat's prompt, prefilled once per chat and
then kept by the cache; and a model offered tools it does not need may call them. Both are measured against today's
offering per activity, on Gemma 4 12B Q4_0, the working model for tool calling, then on every model in a final run:
the prompt's length in tokens, the time to the first token of a new chat, and how often a model calls a tool the
request did not need, on questions that need none. Llama's over-calling is already known (2026-10-25), and its default
may differ.

**On a Mac** a grant holds no folder until the Mac's file checks are built (`../ToolCalling.md` 3.4, v0.21.0 work),
so `request_folder` is not offered there until then.

**Conversations stored with an activity** open as general-purpose chats; a folder they were started in is in their
grant, as it was.

## 6. Mila proposes structure

Tools of the app's own, offered in every chat:

| Tool | Arguments | What it does |
|---|---|---|
| `propose_project` | `name`, `purpose`, `chats` (the chats it would gather) | Offers a project; a yes makes it and moves those chats in |
| `move_to_project` | `project` (by name) | Moves this chat into it: offered, or done and said, by question A |
| `note_understanding` | `summary`, `questions`, `next` | Rewrites the project's *What Mila understands*; runs without asking, since it changes only Mila's own note |
| `link_projects` | `project`, `relation` (section 9) | Offers a link between this chat's project and another (phase 10) |

- **The person may edit before accepting** on the overview's card: the name is a field, and *Make the project* sends
  it as edited. `turn.tool_decision` gains `arguments`, accepted for `propose_project` only.
- **The guidance is in the system prompt**: suggest a project when several chats have become one body of work, once,
  and not again for the same chats if declined; keep the project's understanding current at the end of a turn that
  changed it.
- **Every suggestion, accepted or declined, is an event**, and a project made is undoable from the overview until the
  next change.
- **Measured before on by default**, as section 5.2's tools are.

## 7. Plans and runs

A plan can be worked through again; a run cannot. Each run is a record, and a model's runs differ, since it samples. So
the two are kept apart, and *Run again* makes a new run of the same plan beside the old ones, never a replay.

**Today** a plan is Get Things Done's: the goal is the first message, `propose_plan` gives the steps, `update_step` marks
them, and the chat is the run. Each `update_step` divides the calls between steps, so a run's record is read from the
stored chat with nothing new asked of the model. *Run again* starts a new chat in the project with the plan's goal and
steps already approved: the person's press is the approval `propose_plan` would have asked for.

**When the library has tasks** (`../Agents/AgentTasks.md` section 8): a run is a task, its record the trace, read from
`trace.jsonl` with the page's schema unchanged; the task folder lives in the app's store beside the project, never in a
project's folder (AgentTasks 6); a run ends in AgentTasks 4.2's ways, each a sentence and an event; and a run that asks
the person something (`ask_person`) is a *Needs you* card.

## 8. The store

The session owns it, as it owns conversations today (`../MilaDesktop.md` 6.3), so every window sees the same work:

```
Mila/
    Conversations/<conversation id>.json     as today, plus "project" (id or null) and "process" (id or null)
    Projects/<project id>/
        project.json       version, id, name, purpose, instructions, folders, relations, understanding, created, updated
        history.jsonl      one event a line, appended, never rewritten
        processes.json     each plan: id, goal, steps, version, and its runs (conversation id, started, ended, end)
        creations.json     each creation: id, kind, path or reply, made by (conversation, call, model, time),
                           each write (call, time, SHA-256 of what was written), what was read before it
        findings.json      phase 10
    activity.jsonl         the workspace's events not in a project: chats started, suggestions declined
```

The workspace is this folder: there is one per person (question C).

- **A chat is moved by one field**, never by moving its file.
- **A creation is indexed when a file tool writes it** (`write_file`, `edit_file`), from the call's arguments and
  result, with a digest of the file after the write. A file changed outside Mila is noticed by its digest when its
  drawer opens or the overview is drawn; Mila never watches the disk.
- **A reply the person keeps** is a creation stored in the project's folder as a markdown file, or in the project's
  store when it has no folder.
- **Ids** are made as the conversation store makes them, `project-<UTC time>-<hex>`, and are folder names only when made
  of those characters.
- **Conversations stored before projects** are chats in no project: a missing `project` is null. Nothing is dropped.

## 9. Links and findings (phase 10)

Projects are not a tree. A link joins two projects as *related to*, *depends on*, *derived from*, *part of* or
*supersedes*; *part of* nests nothing in the sidebar. Links are the person's, or Mila's through `link_projects`, offered.

**Findings**: `record_finding` (`text`, `sources`: pages or files read in this chat), allowed by rule since it changes
nothing outside Mila; the person adds, edits and removes findings in the project. A project's understanding and
findings are given to each new chat in it through the system prompt, the newest first, within a share of the context
the app sets and the context ring counts.

## 10. Schema

Additions to `../MilaDesktop.md` 4, all new types or new fields, so `"v"` stays 1.

**Built at phase 1:** `conversations.list` items carry `progress` (`{steps, finished, current}`, or null), `finish` (the
last reply's end, or null) and `wrote` (`[{path, tool}]`, the files written, once each, most recent last, at most 20),
read from the stored conversation by `Desktop.ConversationSummary`.

**Session to page:**

| type | fields | meaning |
|---|---|---|
| `workspace.state` | `name`, `glance` ({`projects`, `active`, `waiting`, `chats`, `chats_loose`}), `needs_you` ([{`kind`, `project`, `conversation_id` or `creation_id`, `text`}]), `suggestions` ([{`id`, `name`, `purpose`, `conversation_ids`}]) | After any change the overview shows; replaces round 1's `home.state` |
| `projects.list` | `items` ([{`id`, `name`, `purpose`, `state`, `updated`, `latest`, `chats`, `made`, `pulse` (fourteen daily counts)}]) | After any change to a project |
| `project.loaded` | `project` ({`id`, `name`, `purpose`, `instructions`, `folders`, `relations`, `understanding`}), `conversations`, `processes`, `creations`, `findings`, `history` (the latest 200 events) | The project now in the pane |
| `activity.list` | `items` ([{`time`, `who`, `kind`, `text`, `project`, `opens`}]), `before` (to page back) | The Activity view |
| `creation.loaded` | `creation` (section 8's fields), `content` (text the page can show, or null), `changed_since` (bool) | The *Made* drawer |
| `run.loaded` | `process_id`, `run` (steps with their calls; the goal, model and end; what it made) | The plan drawer's run |
| `turn.went` | `turn_id`, `place` ({`view`, `id`}), `text` | Mila took the pane somewhere (section 5.2) |

`conversations.list` items and `conversation.loaded` gain `project`; `session.state` gains `project` (the open chat's).

**Page to session:**

| type | fields | meaning |
|---|---|---|
| `project.new` | `name`, `purpose`, `conversation_ids` (optional: move them in) | |
| `project.update` | `id`, any of `name`, `purpose`, `instructions`, `folders` | One change per message |
| `project.open` | `id` | |
| `project.delete` | `id`, `conversations` (`"keep"` or `"delete"`) | Kept chats go to no project; folders are never touched |
| `project.undo` | `id` | Undo a project just made from a suggestion |
| `conversation.move` | `id`, `project` (id or null) | |
| `suggestion.decide` | `id`, `decision` (`"accept"` or `"decline"`), `name` (as edited) | The overview's card |
| `activity.open` | `kind` (or null), `before` | |
| `creation.open` | `id` | |
| `run.open` | `process_id`, `run` (index) | |
| `process.run` | `process_id` | *Run again* |
| `creation.keep` | `turn_id` | Keep a reply as made |

`conversation.new` gains `project` and loses `activity` at phase 3; a stored conversation's `activity` is read and
ignored. `turn.tool_decision` gains `arguments` (section 6) and, for `request_folder`, `folder`: the path the person
picked in the window's picker (`folder.choose`) or one of the project's; its `turn.tool_approval` carries the
project's folders as `folders` to choose among. Where the page shows a view
the person reached by asking, it follows `turn.went`; where they clicked, it sends the matching `*.open` itself.

**A project's folder is a standing grant,** wider than one chosen per chat: offered to every chat in the project. What a
chat may do in it stays its tools' grant (`../ToolCalling.md` 3.4); being in a project never widens a grant.

## 11. Questions

### Where the prototype went against an earlier decision: A and D open for Todd, B and C decided

A. **May Mila file a chat into a project by itself?** The prototype puts a new chat into an existing project and says
   so, with *keep it separate* to undo it. Section 1's rule 3 as round 1 had it, and `../ToolCalling.md`'s gate, say
   Mila proposes and the person decides. Either `move_to_project` is asked about every time, or it runs and is said, as
   `note_understanding` does. Claude's lean: asked the first time for each project, then said, since a wrong filing is
   one undo and an asked filing on every chat is the clutter Todd set out to remove.

B. **No activity cards.** *Decided (Todd, 2026-11-16): "the Mila assistant no longer has cards. It is general purpose
   and works with the user to just do work. The idea is that projects take on structure as they evolve. This is option
   1 then."* Every chat is offered every tool, and each grant is asked for when the work first needs it (section 5.5).
   Not chosen: Mila choosing a chat's tools from its first message, which `../MilaDesktop.md` 3.1 had rejected as "a
   model call before the first token, or a guess".

C. **More than one workspace.** *Decided (Todd, 2026-11-16): "For now: there is only 1 workspace for the user."* The
   sidebar names the workspace and switches nothing, and the store has no level above it (section 8). More than one is
   a later decision, and would be a folder per workspace with the person's settings and the model outside them.

D. **Offers answered in words, the overview's card with buttons.** In the conversation an offer is answered by
   replying; on the overview, Mila's suggestion keeps *Make the project* and *Not now*. Is that the split Todd wants, or
   should the card also be answered by talking?

### Carried, with answers

1. *Is everything in a project?* No: a chat needs no project, and a workspace may be chats alone.
2. *Does a project own one folder?* None or several; a chat that needs one picks among them or adds one.
3. *What a project holds:* chats, instructions, understanding, made, plans, findings, links.
4. *The first run:* makes no project; the setup's *Done* opens the overview, which is then the bar and empty sections.
5. *Conversations stored before projects:* chats in no project.
6. *Deleting a project:* asked, *Keep its chats* the default; folders never touched; its history, plans, index and
   findings go.
7. *Who writes What Mila understands:* Mila, through `note_understanding`, at the end of a turn that changed it; the
   person can edit it. A chat closed without one leaves the last.
8. *Work under way while the person is elsewhere:* needs the session to hold more than one stream over its model, which
   the schema allows and nothing runs; a library question first (phase 11).
9. *Work with the window closed:* out of this design until phase 11.
10. *The release:* Todd's.

## 12. Phases

Each phase is built against the mock first and then the real session. A phase is done when its mock scenarios render as
this document says at 1280x800 and 900 px wide, in light and dark; `node --test` passes; the page makes no network
request; then the same on the real session with Gemma 4 12B Q4_0, and the session's tests pass. The prototype is
updated when a phase changes what it shows. This document is amended in the change that finds it wrong.

1. **The overview's data, from today's records** (built). Each conversation's summary in `conversations.list`; the
   page's sections. Checked: page tests 87/87, session tests 28/28, the real session's list over stdio, the screen over
   the mock at both widths in both themes.
2. **The shell, on today's conversations.** The sidebar (4.1) without projects, the bar (4.2), the overview (4.3) as a
   workspace of chats, *All chats* (4.6), a chat with the pane to itself (4.7), *Activity* (4.8) from what the store can
   say today (chats started, files written, plans approved and stopped), Search (4.11), narrow windows (4.10), and the
   Mila bar with its rising conversation (5.1). Clicking takes the person everywhere; asking does not yet. Until phase 3
   the bar starts what the home screen's box starts today, and the overview keeps the activity cards above *Chats*.
   Replaces `../MilaDesktop.md` 3.3 and 3.4.
3. **General-purpose chats.** Section 5.5: every tool in every chat, `request_folder`, the merged rules, one system
   prompt, stored conversations opened as general chats. The measurement of 5.5 comes first, and decides each model's
   default. Replaces `../MilaDesktop.md` 3.1.
4. **Projects.** The store's `Projects/`; the sidebar's projects; *All projects* (4.4); a project's head, glance and
   chats (4.5); *New project*; moving a chat by its menu; the overview's projects; a project's folders in its chats'
   grant.
5. **Going somewhere by asking.** `open_place` and `find` (5.2), `turn.went`, a request from inside a chat kept out of
   it; measured before on by default.
6. **Mila proposes.** `propose_project`, `move_to_project` by question A, the overview's card and the answered-in-words
   offers (5.3, 5.4), undo.
7. **What Mila understands.** `note_understanding`, the project's summary and *Next* (4.5), given to each chat in the
   project.
8. **Made and where it came from.** The creations index, digests, the *Made* drawer (4.9), *Keep* under a reply.
9. **Plans and runs.** A plan kept as a project's process, the plan drawer, *Run again* (section 7).
10. **Links and findings.** Section 9.
11. **Work while you are elsewhere.** Questions 8 and 9, when a library task and more than one stream exist; a run's
    record moves to the task's trace whenever `AgentTasks.md` phase 5 lands.

## 13. Relationship to other specs

- **`../MilaDesktop.md`** keeps the transcript, the limits, Settings, the session and its schema; its 3.3 and 3.4 are
  replaced at phase 2, its 3.1 at phase 3, and its section 4 gains section 10's messages in the change that builds
  each.
- **`AssistantOnboarding.md`**: the setup ends on the overview.
- **`../Agents/AgentTasks.md`** is where a run's record goes (section 7).
- **`../ToolCalling.md`** owns the gate every offer and placement passes through, and the grant a project's folder is
  offered under.
- **`../Direction.md`** 6.4 is why projects, creations and findings are the app's.
