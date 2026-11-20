# Mila Assistant Interface

**Status:** design, round 3 (2026-11-18), phases 1 to 3 built (`0.21.0-dev+147`, `+149`, `+150`), T1 (`+155`), T2
(`+156`, `+157`), 4 (`+158`), 8 (`+159`) and 5 (`+160`). Round 1, 2026-11-16,
set out projects, processes and creations around a home screen and a sidebar of conversations. Round 2
is the layout Todd reached through a working prototype the same afternoon, iterated with him on the Mac: a sidebar,
a workspace overview, projects that open on their summary, chats that need no project, and Mila as a bar at the foot
of one pane. Todd called it "a good first approximation". Every answer here is Claude's, provisional until Todd settles
it; of four that went against earlier decisions, two are settled (one workspace; general purpose, no activity cards)
and two are open in section 11. Which release each phase lands in is Todd's.

**Round 3** holds everything not yet built to `../Direction/MilaVision.md`, after a review of this design, the code and
the prototype against it. Phase 4 waits for this amendment (Todd, 2026-11-18: "Yes, start on step 1"). Sections 4.12, 5.6
and 5.7 are new; sections 5.2, 5.4, 5.5, 6, 8, 9, 10 and 12 change. **Todd took its recommendations the same day ("Proceed
with your recommendations"; section 11, questions A and D to H), and asked to see the real app in place of a prototype
round.** So what is marked *proposed* below stands as the design, and phase T1 is being built first. The prototype has
not caught up with it.

**Phase 4's decisions** (Todd, 2026-11-19, "Please continue as per your recommendations", taking Claude's lean on each):
*New project* is a sheet until phase 6 (4.1); a project's state is read from its chats' records (4.1); the session is
given the workspace folder, which on macOS moves to `~/Library/Application Support/Mila` (8); one project folder is held
by each of its chats from the start, several are chosen among, and the project's access is where a chat starts (10);
instructions are stored and not shown until phase 7 (8); and the session's and the store's bodies move to implementation
units before any of it is built (12).

**Phase 8's decisions** (Todd, 2026-11-19, "Please continue", taking Claude's lean on each): what was made is read from
each chat's stored calls, with no `creations.json`, so a chat moved between projects takes what it made with it and a chat
in no project has its own (8); each write's call keeps the file's SHA-256, size and modified time after the write, the
model and the plan step then in progress, and a chat stored before has these as *not recorded*; a file is compared with
its last write by size and time first and hashed only when those differ, when its drawer opens and when the overview is
drawn; *what Mila read first* is the calls that read before that file's first write in the same chat, each with whether
it read from outside; the drawer shows a text file's contents up to a limit, markdown drawn as markdown, and the page
names a creation by its chat and path, never by a path alone; a kept reply is kept with its chat in the workspace, and
nothing is written into the person's folders; and *Made* opens from a project's figure and section, Search, a file
written in a chat's transcript, and Activity, where a file written is shown at its call's time.

**Phase 5's decisions** (Todd, 2026-11-20, "Please continue", taking Claude's lean on each; and after the measurement,
"Yes to all and the measure": the prompt's two passages reworded, the file tools offered only once a chat holds a
folder (5.5), and a call its grant refused exposing nothing (`../ToolCalling.md` 3.3)): `open_place` opens the
workspace, *All projects*, *All chats*, Activity, *What Mila can reach*, or a project or chat named in words, and *what
did Mila make here?* opens the project (5.2); names are matched as Search matches them, every word and case folded, an
exact name winning and projects before chats, with up to five returned when several fit; its result says what the place
shows, a list's names and states up to twenty, as the app's own words; in an exposed chat several matches return no
names and the window asks the person to choose, as `request_folder`'s picker does; asked inside a whole chat, the pane
goes there and the chat rises over it, keeping one call and one line; a tool's result may name its own origin call by
call, in `Mila::Agents` (`../ToolCalling.md` 3.3), so a `find_records` match exposes a chat only when it came from an
exposed one; `find_records` reads the person's messages and Mila's replies in every other stored chat, never a call's
result, up to eight matches newest first; a chat's folder is searched by `find_files` and `search_text`, which the
prompt points to, so `find_records` has no folder search of its own; and both tools are measured on Gemma 4 12B Q4_0
before they are offered by default.

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

**Projects are one part of what Mila is for** (`../Direction/MilaVision.md` section 1, decided). Someone may ask Mila to
find an invoice, explain a document or plan a holiday, and most of that should never become a project. So the phases
below are ordered by what a person's digital life needs, and a project is one place work can gather. Whether *Your
Workspace* and *Workspace owner* are the right words is the vision's question 6; nothing here depends on them.

Seven rules follow, and every section below is held to them:

1. **Start anywhere.** The Mila bar at the foot of every screen takes anything, and a chat begins in no project.
2. **Structure grows out of the work.** A chat joins a project when it has substance, by the person's move or Mila's.
3. **The person decides their structure.** Mila suggests a project, a move or a link; how far it may act without a
   yes is open (section 11, question A). Every change is in the workspace's activity, and can be undone.
4. **What was made keeps how it was made.** A file Mila wrote answers what it was for, which chat and plan step made
   it, which model, what Mila read first, every write since, and whether it changed on disk after.
5. **The page shows records; it holds none.** Every view is drawn from what the session stores, so another window, or
   Chat, could show the same work.
6. **What Mila reads cannot steer what Mila does** (*proposed*, section 5.6). A page, a file or a message informs the
   work. Only the person's words and the app's prompts set its course, and ordinary code holds that line.
7. **Nothing leaves this computer unless the person approved that thing** (the vision's local promise, *current
   thinking*; section 5.7). The person sees what is going and where, and can see afterwards what went.

## 2. Where it stands today

- **The shell of section 4 is built on today's conversations** (phase 2, `0.21.0-dev+149`): the sidebar without
  projects, the path bar, the workspace overview as a workspace of chats, *All chats*, a chat with the pane to itself,
  Activity, Search, narrow windows, and the Mila bar with its rising conversation. Every view is drawn from each
  conversation's summary in `conversations.list` (`Desktop.ConversationSummary`; `workspaceOverview`,
  `matchingChats`, `searchResults` and `activityDays` in `state.js`). Clicking takes the person everywhere; asking
  does not yet.
- **Every chat is general purpose** (phase 3, `0.21.0-dev+150`; section 5.5): one prompt, every tool whose work can
  be done on this computer, the network for reading and searching, and no folder until Mila asks for one with
  `request_folder` and the person chooses it in the window's picker. A conversation stored with an activity opens as
  a chat, its folder in its grant.
- **The store** is one JSON file per conversation in `%LOCALAPPDATA%\Mila\Conversations` on Windows
  (`Desktop.ConversationStore.ixx`): its messages, each tool call with its arguments, decision and result, the history
  the model read, and its folder and plan; one stored before phase 3 also carries an activity, which is not read.
- **A chat has a plan when Mila proposes one** (`Desktop.TaskPlan.ixx`): `propose_plan`, approved by the person, and
  `update_step`, allowed by rule.
- **Projects gather chats** (phase 4, `0.21.0-dev+158`): each a record with a name, a purpose, folders and its history,
  in the workspace's own folder beside the conversations; a chat names its project by one field. A project's folder is
  held by its chats at the project's access (section 10).
- **What was made is read from each chat's stored calls** (phase 8, `0.21.0-dev+159`): every file a chat wrote, with the
  file's digest after each write, the model and plan step, and what Mila read first; and each reply the person kept. A
  project's *Made* is its chats'.
- **Mila goes somewhere and finds things when asked** (phase 5, `0.21.0-dev+160`; section 5.2): `open_place` takes the
  pane to a view, a project or a chat, and `find_records` finds what the person's other chats said.
- **Nothing else outlives a conversation**: no instructions or findings shared between chats yet.
- **The library's task** (`../Agents/AgentTasks.md`, agreed 2026-11-03, not built) gives an agent a task carried
  across responses, a plan, a versioned trace of every step, and a task folder holding them.
- **The session holds one conversation's state at a time** (one stream, `../MilaDesktop.md` 4, *Streams*).
- **Against rules 6 and 7, today:** a person can allow `read_web_page` and `web_search` for a whole chat; a link in a
  reply opens on one click with its address shown only on hover; a folder given to a chat is read-write at once; nothing
  records where a chat's content came from; and no view lists what Mila may reach or what left the computer
  (`../Notebooks/MilaVisionDiscussion.md` 8, filed in `../../Issues/Untriaged.md`).

## 3. The concepts

| Concept | The person's word | What it is | Whose |
|---|---|---|---|
| Workspace | *Workspace* | Everything the person has with Mila: projects, chats, activity | The app's store |
| Project | *Project* | A lasting body of work: name, purpose, instructions, folders, its chats and all below | The app's store |
| Chat | *Chat* | The person and Mila talking. In one project, or in none | The app's store, as conversations are today |
| Plan and run | *Plan*, *run* | The steps of a piece of work, and each time they were worked through | The app's store, then the library's task (section 7) |
| Made | *Made* | A file Mila wrote, or a reply the person kept, with where it came from | The app's store, from the calls' records |
| Finding | *Finding* | Something learned, with the sources it rests on | The app's store |
| Event | *Activity* | One thing that happened: never edited, and reduced when what it names is deleted (section 8) | The app's store |
| Understanding | *What Mila understands* | A project's summary, its open questions and its next step, in Mila's words, with what they were drawn from | The app's store, written by Mila (section 6) |
| Source | *What Mila can reach* (working name, section 4.12) | A folder, and later a calendar or a mailbox, with how far Mila may go in it | The app's store |

Three words, chosen against the repository's vocabulary:

- **Workspace, in the window.** It is Todd's word for the top level. In the library *workspace* names scratch
  buffers, and `../Agents/AgentTasks.md` decision 5 kept it out of the task's names for that reason; that rule is about
  identifiers in `Lib/Src`. The app's own code says `Workspace` only where it means this, and never beside the
  library's.
- **Chat, in the window; conversation in the code and the schema.** The messages stay `conversation.*`.
- **Made, not artifact.** *Artifact* keeps its build sense (CLAUDE.md). The spec's own noun is *creation*.

**What is the library's and what is the app's.** Grouping work, keeping projects, creations and findings, and showing
them is the app's: retrieval stores are an application's concern (`../Direction.md` 6.4), and an app adds no capability
to the library (CLAUDE.md). Giving a chat its project's instructions, which the person wrote, is the app writing a system
prompt (`Session::withSystemPrompt`). What Mila wrote or read reaches a chat as data (section 5.6). Kept as the model's
own state, memory would be `Direction.md` 6.4's, and the library's. A run's record is the library's once a task exists
(section 7). **Where labels live is proposed:** the mark of section 5.6 and the gate's use of it in `Mila::Agents`, so a
developer can build the same thing; the store and any connector in the app (question F; the vision's question 4).

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
  **A project's state is read from its chats' records,** with no span of time in it (phase 4): *Waiting on you* when
  one of its chats is under *Needs you* (4.3: a plan stopped short of its end, or a last reply cut short); else
  *Active* while a reply is being written in one of its chats; else *Quiet*. A plan with steps unfinished in a chat
  nobody is replying in is a stopped plan, so it waits on the person rather than counting as active. While the session
  holds one stream (phase 11) at most one project is active.
  **Until phase 6** gives Mila `propose_project`, *New project* opens a sheet: a name, a purpose that may be left
  empty, and *Create*, which sends `project.new`. At phase 6 it asks Mila, and the sheet stays as the way to make one
  without asking.
- **Recent chats**: the four most recent, in any project or none; the open one marked.
- **At the foot**: *Settings* (Command+, or Ctrl+,), the window of `../MilaDesktop.md` 3.7; the model, its name and
  *Model*, which opens what Mila chose (the model, chosen for this computer, the context, thinking, the share of the
  open chat's context); and the person: initials, name, *Workspace owner*, and a menu of Settings, Mila's licence,
  About Mila Assistant and Quit.

As built at phase 2: the model's row ends in the ring of the open chat's share of the context, and while the app
loads, installs or restores, a turning ring and what it is doing take its place. The person is named by the login
name of whoever is signed in to the computer, which the window sends (`about`, section 10). *About Mila Assistant*,
in the menu above *Quit Mila*, opens a dialog with Mila's mark, the version and *Licences*: Mila's, then each the app
carries -- Electron, Chromium, and the session's by NOTICE.md's names (`nlohmann/json`, `CUDA`) -- each opening its
text in the dialog with *Back*, except Chromium's, which opens in the browser (`../MilaDesktop.md` 4, `licence.open`;
as built 2026-11-17, +153).

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
   from*), a note waiting to be accepted (section 5.6), and Mila's suggestion of a project (section 6). Each names its
   project.
4. **Recent projects**: the three most recent as cards, each its name, purpose, state, a strip of its last fourteen
   days of activity, its latest event and its counts (*2 chats · 3 made*); *All N projects* beside the heading.
5. **Chats**: the five most recent not in a project, each its title, the start of Mila's last reply, and when; a
   dashed mark on those Mila sees as belonging together. *All N chats* beside the heading.

Not on it, by Todd's trim the same afternoon: the files made (*Recently made*), and any buttons of suggestions beside
the Mila bar. Phase 1's panels give way to this at phase 2: their data stays, their layout goes.

*As built at phase 8:* a file changed on disk since Mila last wrote it is a card of its own, *Mila noticed*, after
what needs the person, with *Where it came from*; the section is headed *Mila noticed* when it holds nothing else. It
waits on nobody, so it is not counted in *Waiting on you* and turns no project amber. It stays while the file differs
from Mila's last write: nothing yet lets the person say a change was theirs and set it aside (section 11, question I).

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

*As built at phase 4:* the head has the name and purpose, each edited in place, the state and the latest event, and a
menu of *Rename*, *Change what it is for*, *Add a folder* and *Delete project*. Deleting asks, with *Delete, keep its
chats* first and the words *Its folders and the files in them are not touched*. The figures are *Chats*, *Made* (files
Mila wrote in its chats) and *Plans*, the last two opening nothing until phases 8 and 9; *Findings* and the summary come
with phases 10 and 7. Then *Folders* (each with *Read* or *Read and change* and *Remove*, and *Add a folder* through
the window's picker), *Chats* (each with the chat menu, *Move to…* among it), and *History*. The column beside the
contents is not built. *All chats* has the menu's *Move to…* too, listing each project and *No project*.

*As built at phase 8:* *Made* counts the files its chats wrote and the replies kept in them, and opens a section of
them between *Chats* and *History*, the most recent first: each a file's name, or how a kept reply begins, with its chat,
*Changed on disk since* or *No longer on disk* where that is so, and when; each opens its drawer.

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

As built at phase 2, from what the store says today: each chat started, at its `created`; and in a chat the files Mila
wrote and how far its plan came, each at the chat's last change, since the store keeps no time for either. The
filters are *Everything*, *Chats*, *Files* and *Plans*. Its own events, with their times, come with the store's
`activity.jsonl` and `history.jsonl` at phase 4.

*As built at phase 8:* the filter *Files* is *Made*. Each file Mila wrote is an entry at its last write's time
(*Wrote* or *Changed*, opening its drawer), and each reply kept is the person's entry (*Kept a reply* and how it
begins). Files written before calls kept a time are one entry at their chat's last change, as before.

**Activity is also the trace of what left the computer** (*proposed*, section 5.7). A filter, *Sent from this computer*,
lists each page read, search, link opened and network command: exactly what went, where, when, from which chat, and who
allowed it.

### 4.9 Drawers

Over the pane, from the right, closed by Esc or outside it:

- **Made**: the file's name and path, its contents where the page can show them, *Changed on disk after Mila last wrote
  it* when it did, and *Where it came from*: for which project, in which chat, during which plan step, what Mila read
  first, by Mila with which model, and every write.
  *As built at phase 8:* the drawer is a dialog the height of the window, 520 px wide or the window's width. It opens
  from the overview's *Mila noticed* card, a project's *Made*, Search, Activity, and *Where it came from* under each
  file a call wrote in a chat's transcript, shown once the reply is stored. A text file up to 256 KB is shown, markdown
  as markdown; anything else is named only. *Where it came from* gives the project, the chat (opening it), the plan step
  in progress at the first write, the model, when it was made, the latest 20 calls that read before the first write
  (and how many came before those), *You did not write what Mila read here, and it may have shaped what Mila wrote*
  when one of them exposed the chat (5.6), and every write. A kept reply shows its text and what Mila read up to it.
  *Keep* is under each stored reply in a transcript, and *Kept* pressed again lets it go.
- **A plan**: its goal, its steps and their state, its runs and how each ended, *Continue this run* and *Run again*.
- **Settings**, until the app's Settings window takes the place (`../MilaDesktop.md` 3.7).

### 4.10 Narrow windows

Below 900 px the sidebar leaves the window and slides over it from a menu button in the bar, closing once the person
has gone somewhere from it. The overview's figures go two to a row, a project's column of contents goes, and the
conversation may rise higher. Nothing scrolls sideways. A wide window's sidebar does not collapse.

### 4.11 Search

Command+K or Ctrl+K, or *Search* in the sidebar: one box over the pane, projects first, then chats and files, each with
where it is; arrow keys and Enter.

### 4.12 What Mila can reach

*Proposed; the name is a working one, and Todd's to choose.* One place for what Mila has been given, opened from the
person's menu, from Settings and from a chat's head. A new kind of source becomes one more row here, not a screen of its
own (`../Direction/MilaVision.md` section 2).

- **Folders and other sources:** each with the chats or project that hold it, how far Mila may go (*Read*, *Read and
  change*), and *Remove*.
- **This chat:** what the person allowed for it, each with *Remove*, and what it has read from outside (section 5.6).
- **The internet:** that every page and search is asked about, whether commands are on, and whether a search key is set.

A removal applies once any reply being written has ended, since the session takes requests between replies. The view
holds nothing of its own (rule 5). *As built at T1:* the view is a place, *reach*, opened from the person's menu and from
the head of a chat with a folder, and Settings does not yet open it.

## 5. Mila: the bar, and moving through the workspace by talking

### 5.1 The bar and the conversation

**The Mila bar** sits at the foot of every screen, centred in the pane: Mila's mark and one line saying what can be
asked here (*Ask Mila: open a project, see what changed, or start something new*; in a project *Ask Mila about
Calculator app, or go anywhere in your workspace*; with no projects *Ask Mila anything, or open one of your chats*). The
pane fades out beneath it, so it never sits over a card.

**The conversation rises** above the bar when the person sends something, no higher than about half the window, so the
place Mila takes them stays in view. Its head says what it is about (*Talking about your workspace*, *In Calculator
app*, *Not in a project*), with *Open as a whole chat*, *New conversation*, and lower. Esc lowers it; while a reply is
being written, Esc stops the reply first. A chat left while its reply is being written rises over the next place, so
the reply is never out of sight. As built at phase 2 its head names the chat, *New chat* until it is stored, above *Not
in a project*.

**No buttons of suggestions.** Nothing is offered beside the bar or under Mila's replies; the person types (Todd's
trim, 2026-11-16).

### 5.2 Going somewhere by asking

The person can move through the workspace by asking: *show all my projects*, *which projects are waiting on me?*, *open
the calculator project*, *what's next here?*, *what did Mila make here?*, *show all my chats*, *back to my workspace*.
Mila answers in a sentence or two, takes the pane there, and leaves a line in the conversation saying where it went
(*Opened Calculator app*).

**In the app this is the model's, through two tools of the app's own.** The prototype's keyword routing only stands in
for them. They differ in what they can bring into a chat, so they are kept apart (*proposed*):

- **`open_place`** (`place`: *workspace*, *projects*, *chats*, *activity*, or words naming a project or chat) moves the
  pane. The app matches the words against the store and returns the place it opened, or, when several fit, up to five
  to choose from. **No name is listed in the system prompt:** names are the person's data, and one unchanging prompt
  keeps the start every chat reuses (section 5.5, *Measured*). It runs without asking. In an exposed chat (section 5.6)
  a match of several returns no names, and the window asks the person which they meant.
- **`find_records`** (`words`) is the read-only *find it* of the vision's step 9, over past chats first and then a
  folder the person has given. It returns each match, where it is, and the passage that matched, so an answer can cite
  it. It runs without asking in a chat not yet exposed and is asked, with its words shown, in one that is. A match from
  an exposed chat, or from a file, exposes the chat that reads it. (Not `find`, which sits too close to `find_files`.)

Whether models call them well, and do not call them when the person is only talking, is measured on Gemma 4 12B Q4_0
before they are on by default, and on every model in a final run (`ToolCalling.md`'s working model; Llama's
over-calling, 2026-10-25, may make its default differ). `find_records` is also measured on what it is for: found, cited
correctly, and honest when nothing matches.

**As built at phase 5** (`0.21.0-dev+160`), by the decisions at the head of this document: `Desktop.Places` matches the
words and says what a place shows, and `Desktop.RecordSearch` searches the stored chats; both are the session's, in
`Apps/Desktop/Session/Src/Tools/`. A view is named by its own word (`workspace`, `projects`, `chats`, `activity`,
`reach`); a project or chat by its exact name, or else by each word left once *the*, *my*, *project* and *chat* are
dropped being the start of a word in its name. A list's result gives names and whether each waits on the person, the
chat asking never counting as waiting; in an exposed chat it gives only how many, so no list of the person's names
enters a chat that has read from outside. The session sends `turn.went` before the call's result and keeps it on the
stored call, so a chat opened again shows *Opened Calculator app* and goes nowhere. A chat other than the one replying
opens once the reply ends. On a project's page the bar goes on with the conversation that asked Mila to open it, where
a project clicked into begins a new chat as phase 4 decided. `find_records` drops a few common words and needs every
other word in an exchange, a message of the person's and Mila's reply to it searched together, since an answer seldom
repeats the words of its question: searched message by message, Q4_0 on the Mac found *Which tomatoes did we choose for
the south bed?* without its answer and went on searching and opening places. It says per match when its chat had read
from outside, which the model reads as well as the gate.

**Measured** (2026-11-20, Gemma 4 12B Q4_0 on the M1 Pro, its defaults, thinking on; 34 questions asked twice in new
chats of a seeded workspace of three projects and six chats, each chat deleted after its question so no later search
found it; every call the person would be asked about declined; at most five calls a chat; scripts and data
`D:\Claude\phase5`; 68 chats in 1,499 s). On the 16 needing no tool, 30 of 32 made no call; both calls were on *How
should I organise my notes about the garden?*, a project's name (`find_files` twice, `find_records` once). Asked to go
somewhere, 14 of 18 went where asked: *Which of my projects are waiting on me?* opened *All projects* and then every
project and Activity in turn, both times, until the limit stopped it; *Open my chat about the weekend in Lisbon* found
the asking chat by its own title, a defect fixed the same day (the asking chat is never a place); *Open the house bills*
tried `list_folder` in a chat with no folder first, whose refusal marked the chat as having read from outside (the
defect filed at T2, *A tool call that fails still marks the chat*), so the list of projects then gave no names and
the reply ran out of calls. *Open the calculator*, which a project and a chat fit, was not asked about either time: once
Mila said it had no calculator, once it walked from the workspace to the project. Of 10 questions whose answer an
earlier chat held, 8 were found and cited; *Find the electricity invoice from about six months ago* went to the file
tools and `request_folder` first and ran out of calls both times. All 6 with nothing to find said so, with no guess.

**Measured again after the changes it pointed to** (the same day and conditions, 68 chats in 1,283 s): the asking chat
never a place; the two passages saying to open one place and answer from it, and to look in earlier chats before
anywhere else; the file tools offered only once a chat holds a folder (5.5); and a call its grant refused exposing
nothing (`../ToolCalling.md` 3.3). Needing no tool, 30 of 32 made no call: both calls were `find_records` on *How should
I organise my notes about the garden?*, which found nothing and answered, a read-only search the person sees and that
moves nothing. Going somewhere, 18 of 18, with *Open the house bills* searching the chats and opening *All projects*
before the project both times. Both choices between a project and a chat that fit were asked about. Finding, 10 of 10
by reading each answer (the script's literal check passed 8: one answer was cut where the record ends, one wrote *Belém*
for *Belem*). Nothing to find, 6 of 6. On this, both tools stay in every chat's prompt.

**A request to go somewhere made from the bar outside a chat** belongs to the bar's own conversation. **Made inside a
chat, it stays in that chat** as one call and one line (*proposed*, replacing "it begins a conversation of its own"): the
session holds one conversation at a time, so a second would evict the chat's state, and returning would prefill the chat
again at a cost nobody has measured. It can move out when several streams exist (phase 11).

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

**The model never decides that an offer was accepted** (*proposed*):

- **An offer is a record the session holds,** made by a tool call, and the window draws it from that call's arguments.
  The person is answering what the app showed, never what the model's sentence implied.
- **App code reads the reply.** A reply that is only an acceptance (*yes*, *ok*, *sure*, *please do*) takes the offer. A
  refusal declines it. Anything else closes it undecided and goes to Mila as an ordinary message.
- **Only what one undo reverses** can be taken in words: making a project from a suggestion, moving a chat. Nothing that
  writes a file, sends anything or widens what Mila can reach.
- **An offer from an exposed chat** (section 5.6) is answered in the window, with its details in view, never in words.

### 5.5 A general-purpose chat

Decided by Todd, 2026-11-16 (question B): no activity cards. A chat is not started as a kind of work; it is work, and
what it needs is asked for when it first needs it. This replaces the activities of `../MilaDesktop.md` 3.1, which
decided a conversation's tools, their grant, its system prompt and its limits when it began.

**Every chat is offered every tool the app has**: the file tools (`list_folder`, `read_file`, `find_files`,
`search_text`, `write_file`, `edit_file`), web reading, web search when the person has given a search key, the plan (`propose_plan`, `update_step`),
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
  **A folder starts read-only (question G),** as the vision's ladder has a source start. The first call that would change
  a file says so in its own question: allowing it also lets Mila change files in that folder for the rest of the chat.
  Declined, the folder stays readable. A chat reopened later holds its folder read-only until the person allows a change
  again. Commands need that same allowance, since they run with write access to the folder.
- **The network, for reading public pages and searching.** Every page read and every search is asked (section 5.7).
- **One rule:** `update_step` runs without asking. Every other call is asked, and a command every time
  (`../ContainedPrograms.md` 5).

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

**Measured** (2026-11-17, Gemma 4 12B Q4_0 on the M1 Pro, its defaults, thinking on; 18 questions asked twice in new
chats, every call declined so none ran; scripts and data `D:\Claude\phase3`). On the 13 questions needing no tool,
three of them near the tools' subjects: Ask & Explore's prompt with no tools, 0 calls in 26; the draft prompt offered
every tool but search and commands, 0 in 26; with the Mac's tools alone (plan and page reading), 0 in 26; Get Things
Done's prompt, 2 in 13 (`list_folder` for *Who wrote Pride and Prejudice?*, and a malformed plan). The prompt and a
short question came to about 73 tokens with no tools, 600 with the Mac's, 1,340 with every tool (*arithmetic*, the
context after a one-round reply less its tokens); the first chat after a load took 1.1 s to its first token with no
tools and 9.3 s with every tool, and every later chat 0.7 to 0.75 s in all four, as the session keeps the prompt's start
from chat to chat. Given a page's address, every-tool read it 6 times in 6; the Mac's set proposed a plan first 5 times
in 6, until the prompt said *a question that one page or one tool call answers needs no plan*, then read it 6 in 6, still
0 in 26 needing none. Get Things Done's prompt ran one reply into a loop of 132,843 tokens (`Issues/Untriaged.md`).
Not yet measured: `request_folder`'s own pull on questions needing no files, and every model's final run; until then
every model is offered the same.

**As built at phase 3** (`0.21.0-dev+150`): `Apps/Desktop/Prompts/general.json` is the one prompt, its system prompt
written as passages, each kept only while the tool it names is offered and one only while web search is not
(`Desktop.PromptFile`, `Desktop.PromptPart`), since a prompt naming a tool not offered draws calls of it. A chat holds
one folder: `request_folder` asked again in a chat that has one is answered with it, unasked. A file tool called in a
chat with no folder is not asked about, since its grant reaches nothing: it fails, and Mila asks for a folder, as it did
on the PC with Q4_0 (it listed the folder, read the README and answered). The plan's `update_step` runs by rule; every
other call is asked. *Changed at phase 5* (Todd, 2026-11-20): a tool that works in a folder is offered only once the
chat holds one, as this section's own rule for a tool that cannot do its work says, since phase 5's measurement found
models calling them in chats with no folder; until then the chat has `request_folder`, and the chat's prompt changes
when it is given a folder, which reads the conversation again once from where the tools are listed.

**On every platform.** The grant's file checks exist on Windows, on macOS (`0.21.0-dev+152`) and on Linux (`+153`, tested
in WSL, where the window cannot run, so the window on Linux is unverified; `../ToolCalling.md` 3.4). A computer where the
library cannot grant a folder is offered neither `request_folder` nor a tool that needs one, rather than tools that can
only fail. The session finds this by asking the library for a grant of one.

**Conversations stored with an activity** open as general-purpose chats; a folder they were started in is in their
grant, as it was. One that cannot be granted opens without it and says so.

### 5.6 What content may cause

*Decided (Todd, 2026-11-18, questions E to H as section 11 records); the mark and the gate's part built as phase T2
(`0.21.0-dev+156`), in `Mila::Agents` (`../ToolCalling.md` 3.3, *What a result may cause*).* This is as much of
`../Direction/MilaVision.md` section 8 as the phases below need. The vision's wider questions stay there.

**A chat is *exposed* once anything enters it that may not direct Mila:** a page, search results, a file or a file's
name, a command's output, an attachment, a match from another chat, or a marked note. The person's words and the app's
own prompts and answers (`request_folder`'s, the plan's, a refusal) do not expose it. The person's own files count until
question E is settled, because the cost of counting them is more questions and the cost of not counting them could be
harm.

**The mark is on the chat, not on each passage.** What the model writes after reading something takes the least trust
of what it read, so everything after the first exposure is exposed anyway. A mark on the chat survives storage,
compaction, a restart and a change of model by construction, which is the vision's invariant 3. The list behind it says
what was read, from where and when. The price is coarseness: the mark stays until the chat ends, and a new chat starts
clear. Tracking passages can come later if the questions prove too many.

**What the mark changes,** in ordinary code at the gate (the vision's invariant 4):

1. **A call that writes, sends or runs is asked every time.** No rule and no *for this conversation* covers it, and one
   given earlier stops applying when the chat becomes exposed. Reads inside a granted folder may stay allowed, since what
   they bring in cannot leave without its own approval.
2. **The question says why:** *This chat has read* and its latest sources. For an address or search words it also says
   when they do not appear in the person's own messages, by literal comparison in the app.
3. **The app's own tools follow the table below.**
4. **What the chat writes to a lasting record carries the mark** (section 8).

| Tool | In a chat not exposed | In an exposed chat |
|---|---|---|
| `open_place` | Runs | Runs; several matches go to the person, not the model |
| `find_records` | Runs, read-only | Asked, its words shown |
| `propose_project` | Offered, answered in words or on the card | Offered, answered in the window only |
| `move_to_project` | By question A | Asked |
| `note_understanding` | Runs; the note can reach new chats | Runs; the note waits for the person |
| `record_finding` | Runs; its sources are checked | Runs, sources checked; the finding waits for the person |
| `update_step` | Runs | Runs |

**What reaches a new chat.** Only the person's own project instructions go in the system prompt. Understanding and
findings are placed after the person's first message, as the result of a call the app makes, quoted and named as notes
with their sources. So they are read as data, and the system prompt stays the same for every chat, which keeps the
prompt's start reusable. How each family's template takes this is measured first; if the library needs a way to place
such a result, that is a proposal to Todd. A note from an exposed chat **waits** on the project's page, in full and with
where it came from, until the person accepts it. Accepting lets it reach new chats and removes no label: a chat given a
marked note starts exposed.

**What this does not do.** A label does not stop a model from following what it read. The protection is that the
consequences are held by code: what leaves, what changes and what lasts each pass a check the model cannot talk its way
past. Text the person pastes counts as their words, because the app cannot tell it apart.

**Checked by** tests with a scripted model that obeys a hostile page, needing no real model: it cannot write, send or
run without a question; an earlier allowance stops covering it; its note reaches no new chat until accepted; the mark
survives storage, compaction and a change of model. Each test fails with its check removed. How many extra questions a
real folder task costs is counted before question E is settled.

### 5.7 What leaves this computer

*From the vision's section 3 and steps 2 to 4; taken by Todd 2026-11-18, and built as phase T1 (`0.21.0-dev+155`).*
Nothing outward is covered by an earlier answer. *For this conversation* is never offered for a tool that uses the network, which the gate can tell
from the tool's declared needs, so a developer's program gets the same rule.

| Path | The person sees, before it goes | Rule |
|---|---|---|
| `read_web_page` | The whole address | Asked each time. A redirect to another host, or from `https` to `http`, does not follow: it returns to the model, and going there is a new call and a new question. `http` is said to be unencrypted |
| `web_search` | The exact words, and that they go to the search provider | Asked each time |
| `run_command` with the network | The command, and that it can send anything from the folder to any address | Asked each time. The question cannot show what will be sent (`../../Issues/Untriaged.md`) |
| A link in a reply | The whole address, host first, or a `mailto:` address's recipient, subject and body, in full and never shortened | A click opens a sheet in the page with *Open* and *Cancel*, every time |
| Listing and installing models | `AssistantOnboarding.md` 4.2 | Only on the person's press |
| The window's proxy discovery | Nothing, because it stops | The window uses no network and looks for no proxy |

**Each thing sent is recorded** with what went, where, when and from which chat, and appears in Activity (section 4.8):
each call's stored record now has its time, and a link opened is kept with the chat. The record is private data like any
other, and is deleted with its chat.

**Checked by** tests on the gate, the web reader and the summary of what was sent, each with a scripted transport or
stored record, and by the page's tests of the sheet's and the question's words. The sheet, the question and the first-run
press are checked in the window on Windows and macOS; on Linux the code and its tests run in WSL, and the window is
unverified. **Not built yet:** the conformance test of the vision's step 4, which drives a whole turn with no approval
through a scripted transport.

**Outside what Mila can see:** a file written into a folder a sync client watches, and text copied to a clipboard the
system syncs (`../Notebooks/MilaVisionDiscussion.md` 8.4). Public wording of the promise must not claim otherwise.

## 6. Mila proposes structure

Tools of the app's own, offered in every chat:

| Tool | Arguments | What it does |
|---|---|---|
| `propose_project` | `name`, `purpose`, `chats` (the chats it would gather) | Offers a project; a yes makes it and moves those chats in |
| `move_to_project` | `project` (by name) | Moves this chat into it: offered, or done and said, by question A |
| `note_understanding` | `summary`, `questions`, `next` | Rewrites the project's *What Mila understands*. It runs without asking, and from an exposed chat the new note waits for the person (section 5.6) |

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

**Today** a plan belongs to its chat: the goal is the person's request, `propose_plan` gives the steps, `update_step` marks
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
    Conversations/<conversation id>.json     as today, plus "project" (id or null), "process" (id or null),
                                             "exposure" (what it read from outside) and each call's time
    Projects/<project id>/
        project.json       version, id, name, purpose, instructions, folders (each with its access), created, updated
        understanding.json the current note and any waiting, each a record
        history.jsonl      one event a line
        processes.json     each plan: id, goal, steps, version, and its runs (conversation id, started, ended, end)
        findings.json      each finding, a record (phase 10)
    activity.jsonl         the workspace's events not in a project: chats started, suggestions declined
```

The workspace is this folder: there is one per person (question C).

**Where it is, and how the session is given it** (phase 4). The session takes the workspace folder, `--workspace <dir>`,
in place of the conversations folder it took before (`--conversations`), and keeps `settings.json` at its top as before,
so nothing on disk moves on Windows or Linux:

| Platform | The workspace folder | Its folders |
|---|---|---|
| Windows | `%LOCALAPPDATA%\Mila` | `Conversations`, `Projects` |
| macOS | `~/Library/Application Support/Mila` | `Conversations`, `Projects` |
| Linux | `$XDG_DATA_HOME/mila`, else `~/.local/share/mila` | `conversations`, `projects` |

macOS kept conversations in Linux's place until phase 4; no released Mila wrote there, so nothing is moved from it.

**A project's instructions are stored from phase 4 and not shown or given to a chat until phase 7,** which brings the
project's understanding into its chats in the same change, so the cost of what a chat is given at its start is measured
once for both.

**Every lasting thing Mila derives is a record of one shape** (*proposed*, the vision's section 6), so `find_records`, the
drawers and deletion read one form:

```
{ "id", "kind", "made",
  "source":       { "kind": "person|app|model|web|search|file|command|chat", "ref", "when", "author" or null },
  "may_direct":   false,
  "may_leave":    "ask",
  "derived_from": [ the records, chats or calls it rests on ],
  "exposed":      true or false,
  ...its own fields }
```

**One layer owns the files.** Conversations and everything above are read and written through one store module in the
app, and nothing else opens these files. It deals in whole records and never puts content in a file's or folder's name,
so encrypting them changes one place. *Checked at T2 (`0.21.0-dev+156`):* conversations are `Desktop.ConversationStore`'s
alone, which loads and saves whole records and names a file by its id only; `Projects/` and `activity.jsonl` join it when
phase 4 makes them. `settings.json` holds no conversation content and stays the settings file's. Each conversation now
keeps `exposure`, what it read from outside (5.6), and each stored call `exposes`. The search key is in the operating
system's credential store on Windows, macOS and Linux (`../MilaDesktop.md` 3.7).

**Encrypting what is at rest** (*proposed*, Claude, 2026-11-19; the vision's step 5; nothing built). What it protects
against: another account on the computer, a disk or backup read without the person's sign-in, and a sync client or
cloud backup carrying the folder away. What it does not: anything running as the person while they are signed in,
which can ask for the key as Mila does.

- **One key per workspace,** 256 random bits, made on first use, kept in the same credential store as the search key
  (Windows Credential Manager, the macOS login keychain, the Secret Service on Linux), never in the workspace folder.
- **Each file sealed whole,** with an authenticated cipher (AES-256-GCM or XChaCha20-Poly1305, chosen by what each
  platform provides without a new dependency: CNG on Windows, CommonCrypto or CryptoKit on macOS, and on Linux the
  library that also serves curl), a fresh nonce per write, and the file's path in the store as associated data, so a
  file moved or swapped into another's place fails to open. The store layer above is the only place this happens.
- **A file that fails to open is kept and named,** never deleted or overwritten, and the window says so.
- **No key, no store:** a Linux computer with no keyring cannot keep the key, so conversations there are not stored
  until one exists, and the window says why. This needs Todd's decision, since it changes what Linux without a desktop
  session can do.
- **Moving in:** files written before encryption are sealed one by one on the first start after it, each written
  sealed beside the plain file and renamed over it, as saving already is.
- **Open for Todd:** the cipher; whether the key may be exported for the person's own backup; what Linux with no
  keyring does; and whether *Search* (4.11) reads sealed files whole or keeps an index, which would need sealing too.

**Deleting is designed in** (*proposed*; the vision's *see, trace, delete*):

- **Deleting a chat or source removes what was derived from it.** A finding left with no source goes. A finding with
  others loses that one.
- **An understanding is cleared** when something it was drawn from is deleted, because a paragraph cannot be unmixed. Mila
  writes it again at the next turn that needs it.
- **Events are appended in ordinary work.** A deletion rewrites the file: an event that names what went keeps its kind and
  time, and loses the name and text.
- **A file in the person's folder is never touched.** The question before deleting says so, and the file's record goes.

- **A chat is moved by one field**, never by moving its file.
- **A creation is read from the call that wrote it** (`write_file`, `edit_file`; phase 8's decisions, no
  `creations.json`): the stored call keeps `written` (`location`, `sha256`, `size`, `modified`, the file as the write left
  it), `model` and `step`. A file changed outside Mila is noticed when its drawer opens or the conversations are listed,
  by size and time and then its digest; Mila never watches the disk.
- **A reply the person keeps** stays with its chat: the stored message gains `kept`, the time. Nothing is written into
  the person's folders.
- **Ids** are made as the conversation store makes them, `project-<UTC time>-<hex>`, and are folder names only when made
  of those characters.
- **Conversations stored before projects** are chats in no project: a missing `project` is null. Nothing is dropped.

## 9. Findings, and links if they are ever needed (phase 10)

**Findings**: `record_finding` (`text`, `sources`: pages or files read in this chat) runs without asking, since it changes
nothing outside Mila, and the app refuses a source the chat's stored calls do not show it read. From an exposed chat the
finding waits for the person (section 5.6). The person adds, edits and removes findings in the project. A project's
understanding and findings reach each new chat as section 5.6 says, the newest first, within a share of the context the
app sets and the context ring counts.

**Links between projects are held back** (*proposed*). Round 2 had a link join two projects as *related to*, *depends on*,
*derived from*, *part of* or *supersedes*, through `link_projects`. The vision puts a graph of relationships after
records that already exist, because a local model's errors would become stored facts. So no link, tool or `relations`
field is built until real use asks for one, and then it points at its evidence and can be deleted.

## 10. Schema

Additions to `../MilaDesktop.md` 4, all new types or new fields, so `"v"` stays 1.

**Built at phase 1:** `conversations.list` items carry `progress` (`{steps, finished, current}`, or null), `finish` (the
last reply's end, or null) and `wrote` (`[{path, tool}]`, the files written, once each, most recent last, at most 20),
read from the stored conversation by `Desktop.ConversationSummary`.

**Built at phase 2:** `conversations.list` items carry `created` (when the conversation was made) and `reply_start` (the
start of the last reply, on one line, never cut inside a character, or null). The window sends the page `about`
(`user_name`, `version`) when the page is ready (`../MilaDesktop.md` 4.1).

**Session to page:**

| type | fields | meaning |
|---|---|---|
| `workspace.state` | `name`, `glance` ({`projects`, `active`, `waiting`, `chats`, `chats_loose`}), `needs_you` ([{`kind`, `project`, `conversation_id` or `creation_id`, `text`}]), `suggestions` ([{`id`, `name`, `purpose`, `conversation_ids`}]) | After any change the overview shows; replaces round 1's `home.state` |
| `projects.list` | `items` ([{`id`, `name`, `purpose`, `state`, `updated`, `latest`, `chats`, `made`, `pulse` (fourteen daily counts)}]) | After any change to a project |
| `project.loaded` | `project` ({`id`, `name`, `purpose`, `instructions`, `folders`, `understanding`, `waiting`}), `conversations`, `processes`, `creations`, `findings`, `history` (the latest 200 events) | The project now in the pane |
| `activity.list` | `items` ([{`time`, `who`, `kind`, `text`, `project`, `opens`}]; `kind` includes `sent`), `before` (to page back) | The Activity view |
| `creation.loaded` | `creation` (section 8's fields), `content` (text the page can show, or null), `changed_since` (bool) | The *Made* drawer |
| `run.loaded` | `process_id`, `run` (steps with their calls; the goal, model and end; what it made) | The plan drawer's run |
| `turn.went` | `turn_id`, `place` ({`view`, `id`}), `text` | Mila took the pane somewhere (section 5.2) |

`conversations.list` items and `conversation.loaded` gain `project` and `exposed`; `session.state` gains `project` (the
open chat's). `turn.tool_approval` gains `exposure` (the latest sources, or empty) and `not_from_person` (the arguments
that do not appear in the person's messages).

**Built at T1:** `session.state` carries `folder_access` (`"read"`, `"read_write"` or null) and `allowed_for_chat` (tool
names). `conversations.list` items carry `sent` ([{`kind`: `page`, `search`, `command` or `link`, `what`, `where`, `time`}],
oldest first, the latest 50). `turn.tool_approval` carries `changes_folder` (the folder allowing this call opens to
changes, or null), and a tool that uses the network never lists `allow_conversation`. A stored call has `time`, and a
stored conversation `links_opened` ([{`url`, `time`}]).

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
| `note.decide` | `project`, `id`, `decision` (`"accept"` or `"decline"`) | A note or finding that was waiting |
| `reach.remove` | `kind` (`"folder"` with the chat's `id`, or `"allowed"` with the tool's `name`) | Built at T1. Applies once any reply being written has ended |
| `conversation.link_opened` | `url` | Built at T1. The person opened this link from the open chat |

**Built at phase 3:** `conversation.new` carries nothing, and an `activity` or `folder` sent is not read; a stored
conversation's `activity` is not read. `session.state` loses `activities` and `activity`, and `conversations.list` and
`conversation.loaded` lose `activity`. `turn.tool_decision` gains `folder` for `request_folder` allowed: the path the
person picked in the window's picker (`folder.choose`); one allowing it with no folder is ignored. Its
`turn.tool_approval` offers `allow_once` and `deny`, since a folder is chosen each time it is asked for.

`conversation.new` gains `project` at phase 4. `turn.tool_decision` gains `arguments` (section 6), and its `folder` may
be one of the project's, which `request_folder`'s `turn.tool_approval` carries as `folders` to choose among, with the
access the project's folder is held at. Where the page shows a view
the person reached by asking, it follows `turn.went`; where they clicked, it sends the matching `*.open` itself.

**A project's folder is a standing grant,** wider than one chosen per chat: offered to every chat in the project. What a
chat may do in it stays its tools' grant (`../ToolCalling.md` 3.4); being in a project never widens a grant. *Proposed:*
it is the first **source** of section 4.12, listed there, read-only unless the person chose *Read and change*, and
removable at any time. How far standing access goes beyond a folder the person named is the vision's question 3.

**As decided for phase 4.** A chat still holds one folder (5.5, as built at phase 3). A chat in a project with one
folder holds it from its start; in a project with several, `request_folder` offers them, and a folder chosen outside
them is that chat's alone and is not added to the project. The project's access is where its chats start: *Read*, and a
chat's first write asks to change files for the rest of that chat, as in a chat of its own; *Read and change*, and its
chats start able to change files, since the person chose that on the project. A chat moved into a project keeps the
folder it holds. Removing a folder from the project takes it from the chats that hold it through the project, once any
reply being written has ended. Changing a folder to *Read* narrows the open chat at once; changing it to *Read and
change* reaches a chat from its next start, so nothing widens under a chat already running. A chat reopened in a project
holds the project's folder as the project has it then.

**Built at phase 4:**

- **Stored:** a conversation carries `project` (an id, or null) and `folder_from_project`. `Projects/<id>/project.json`
  holds `v`, `id`, `name`, `purpose`, `instructions`, `folders` (`[{path, access}]`), `created`, `updated`; its
  `history.jsonl` and the workspace's `activity.jsonl` hold events `{time, who, kind}` with any of `project`, `from`,
  `conversation`, `name`, `path`, `access`. The kinds are `project.made`, `project.renamed`, `project.purpose`,
  `project.instructions`, `project.folder_added`, `project.folder_access`, `project.folder_removed`, `project.deleted`
  and `chat.moved`. An event goes in the history of each stored project it names as `project` or `from`, else in
  `activity.jsonl`. A deletion rewrites each file to forget the id, and the name and folder that came with it.
- **Session to page:** `projects.list` carries each project's record, and the page works out its state, chats, what
  was made and its fourteen days from the records it already has, so `projects.list` is not the table's richer item.
  `activity.list` carries the events, newest first, at most 200, and `before` the time to page back from, or null.
  `project.made` (`id`) follows a `project.new`, so the page opens the project. `session.state` gains `project` and
  `folder_from_project`; `conversations.list` items gain `project` and `folder_from_project`, and `conversation.loaded`
  gains `project`. `request_folder`'s `turn.tool_approval` carries `folders`, the project's, to choose among.
- **Page to session:** `project.new`, `project.update` (one change a message), `project.delete`, `conversation.move`,
  `conversation.new` with `project`, and `activity.open` (`before`). Not built: `project.open` and `project.loaded`,
  since the page draws a project from the lists it has; `project.undo`, `suggestion.decide` and `note.decide` (phases 6
  and 7); `creation.*` and `run.*` (phases 8 and 9).

**Built at phase 5:**

- **Stored:** an `open_place` call carries `went` ({`place`, `text`}).
- **Session to page:** `turn.went` (`turn_id`, `call_id`, `place` {`view`, `id`}, `text`), sent before the call's
  result. `turn.tool_approval` carries `places` ([{`view`, `id`, `name`}]) for `open_place` in an exposed chat when
  several fit.
- **Page to session:** `turn.tool_decision` carries `place` ({`view`, `id`}, one of the approval's) when it allows such
  a call.

**Built at phase 8:**

- **Stored:** a write's call carries `written` ({`location`, `sha256`, `size`, `modified`}, or null when the file could
  not be read after it), `model` and `step` (the plan step in progress, or null); an assistant message carries `model`,
  and `kept` (a time) once the person keeps it.
- **Session to page:** `conversations.list` items' `wrote` entries gain `time` (the last write's) and `disk` (`"same"`,
  `"changed"`, `"missing"`, or null), and items gain `kept` ([{`message`, `time`, `start`}]). `turn.end` gains `message`.
  `creation.loaded` is `creation`, `content`, `content_kind` and `disk`, as `../MilaDesktop.md` 4.1 lists them.
- **Page to session:** `creation.open` names a creation by `conversation_id` and `path`, or `conversation_id` and
  `message`, not by an `id`, since a creation has no record of its own; `creation.keep` takes `conversation_id`,
  `message` and `keep`, the reply being a stored message rather than a turn the session still holds.

## 11. Questions

### All decided. A and D to H: Todd, 2026-11-18, "Proceed with your recommendations", taking each lean below

A. **May Mila file a chat into a project by itself?** The prototype puts a new chat into an existing project and says
   so, with *keep it separate* to undo it. Section 1's rule 3 as round 1 had it, and `../ToolCalling.md`'s gate, say
   Mila proposes and the person decides. Either `move_to_project` is asked about every time, or it runs and is said, as
   `note_understanding` does. Claude's lean: asked the first time for each project, then said, since a wrong filing is
   one undo and an asked filing on every chat is the clutter Todd set out to remove. Always asked in an exposed chat
   (section 5.6).

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
   should the card also be answered by talking? Claude's lean: keep both, under section 5.4's rules.

E. **Does reading the person's own files expose a chat?** The vision leaves this open (section 8; its question 2).
   Claude's lean: yes to begin with, since a file may have come from anywhere. The cost is real: after the first read,
   each write in that chat is asked. Count the questions in a real folder task before settling it.

F. **Do the mark and its enforcement belong in `Mila::Agents`,** with the store and connectors in the app? Claude's lean:
   yes, so the gate can read the mark and a developer can build the same thing. This amends `../Direction.md` 6.4 (the
   vision's question 4).

G. **Does a folder start read-only?** Claude's lean: yes, with changing files as a separate, visible step (section 5.5).

H. **Does a note from an exposed chat wait for the person?** Claude's lean: yes. Otherwise a hostile page could shape
   every later chat in a project. The cost is one more thing under *Needs you* after research.

### Open

I. **May the person set aside a change on disk?** (Claude, phase 8.) A file Mila wrote and the person then edited stays
   under *Mila noticed* for as long as it differs from Mila's last write, though the change was theirs. Claude's lean: a
   *That was me* on the card and in the drawer, stored on the chat as the digest the person accepted, so a later change
   is noticed again. Not built until Todd decides.

### Carried, with answers

1. *Is everything in a project?* No: a chat needs no project, and a workspace may be chats alone.
2. *Does a project own one folder?* None or several; a chat that needs one picks among them or adds one.
3. *What a project holds:* chats, instructions, understanding, made, plans, findings.
4. *The first run:* makes no project; the setup's *Done* opens the overview, which is then the bar and empty sections.
5. *Conversations stored before projects:* chats in no project.
6. *Deleting a project:* asked, *Keep its chats* the default; folders never touched; its history, plans, index and
   findings go.
7. *Who writes What Mila understands:* Mila, through `note_understanding`, at the end of a turn that changed it; the
   person can edit it. A chat closed without one leaves the last. From an exposed chat it waits (question H).
8. *Work under way while the person is elsewhere:* needs the session to hold more than one stream over its model, which
   the schema allows and nothing runs; a library question first (phase 11).
9. *Work with the window closed:* out of this design until phase 11.
10. *The release:* Todd's.

## 12. Phases

Each phase is built against the mock first and then the real session. A phase is done when its mock scenarios render as
this document says at 1280x800 and 900 px wide, in light and dark; `node --test` passes; the page makes no network
request; then the same on the real session with Gemma 4 12B Q4_0, and the session's tests pass. Every feature lands on
Windows, macOS and Linux, with the window on Linux stated as unverified. The prototype is updated when a phase changes
what it shows. This document is amended in the change that finds it wrong.

**The phases keep their numbers, since commits and other documents cite them, and are not built in numeric order.** The
order after phase 3 is **T1, T2, 4, 8, 5, 6, 7, 9, 10, 11**. Each of T1 and T2 is small enough to finish alone.

1. **The overview's data, from today's records** (built). Each conversation's summary in `conversations.list`; the
   page's sections. Checked: page tests 87/87, session tests 28/28, the real session's list over stdio, the screen over
   the mock at both widths in both themes.
2. **The shell, on today's conversations.** The sidebar (4.1) without projects, the bar (4.2), the overview (4.3) as a
   workspace of chats, *All chats* (4.6), a chat with the pane to itself (4.7), *Activity* (4.8) from what the store can
   say today (chats started, files written, plans approved and stopped), Search (4.11), narrow windows (4.10), and the
   Mila bar with its rising conversation (5.1). Clicking takes the person everywhere; asking does not yet. Replaces
   `../MilaDesktop.md` 3.3 and 3.4. Built (`0.21.0-dev+148`, `+149`). The activity cards went here rather than at phase
   3 (Todd, 2026-11-16: "The cards are gone"): until phase 3 the bar starts *Ask & Explore*, and a conversation stored
   in another activity opens with its tools and folder. Checked: page tests 91/91, window tests 5/5, session tests
   31/31 (`x64-claude-verify`, every target built); the mock at 1280x800, 900 and 860 px in light and dark, with no
   console error and nothing scrolling sideways; the real session's `conversations.list` over stdio after a turn on
   Gemma 4 12B FP4, and the window on the real session with the same model, in a scratch store: setup, a turn risen
   over the overview, lowered, Activity, and the chat opened whole. Not yet: Todd's look on the Mac.
3. **General-purpose chats.** Section 5.5: every tool in every chat, `request_folder`, the merged rules, one system
   prompt, stored conversations opened as general chats. The measurement of 5.5 comes first, and decides each model's
   default. Replaces `../MilaDesktop.md` 3.1. Built (`0.21.0-dev+150`) after 5.5's measurement on Gemma 4 12B Q4_0.
   Checked: session tests 35/35, page tests 92/92 (mock scenario 14, Mila asks for a folder), window tests 5/5, every
   target built in `x64-claude-verify`; scenario 14 over the mock at 1280 and 900 px in light and dark; the real session
   over stdio on Q4_0 in a scratch store: a chat asked about a project, Mila asked for a folder, was given one, listed
   and read it, and answered. On the Mac (packaged, Q4_0) the same question was answered without files, since no folder
   could be granted there then. *Built 2026-11-17 (`0.21.0-dev+152`):* the grant's macOS file checks
   (`../ToolCalling.md` 3.4); the session on the Mac, staged, over stdio on Q4_0: asked what was in its project folder,
   Mila called `request_folder`, was given a folder, listed it and read `notes.txt`. **Not done:** the same through
   the window's open panel on the Mac, and a chat reopened with its folder after the app restarts (`../ToolCalling.md`
   3.4, *Not measured*); `request_folder` measured on questions needing no files; every model's final run; the window
   on the real session.

The rest, in the proposed order:

- **The prototype's next round** is set aside: Todd chose to judge T1 in the real app on the Mac (2026-11-18). The
  prototype's example work and chat kinds are still to be brought up to this design when it is next used.
- **T1. What leaves, and what Mila can reach** (the vision's steps 2 to 4). Section 5.7; folders read-first (5.5); section
  4.12 for folders and what a chat was allowed; each call's time stored, and *Sent from this computer* read from the
  stored calls. *Built 2026-11-18 (`0.21.0-dev+155`).* Checked: page tests 103/103, window tests 9/9 and session tests 36/36 on
  Windows and macOS; the gate, web reader and client tests on both; every screen over the mock in the window on Windows,
  whose network log held no proxy lookup in 25 s where it held four before; and on the Mac, through the packaged session
  on Gemma 4 12B Q4_0, a folder given read-only, the first write asking to change it and then landing, and a page read
  offered only *Allow once* and *Deny*. On Linux nothing was run for this phase. **Not done:** the conformance test through a whole turn, a question
  that says which chat content led to it (T2), and asking before a gated model's download uses the person's account.
- **T2. The mark and one store** (the vision's steps 5 and 7). The source of each tool result and section 5.6's mark
  through the gate, storage and compaction; the store layer of section 8 with conversations behind it; the search key in
  the credential store; the encryption design written. Done when section 5.6's tests pass. *Built 2026-11-19
  (`0.21.0-dev+156`):* the mark in `Mila::Agents` (`../ToolCalling.md` 3.3); the session's own tools declared as its own
  answers; `exposure` and each call's `exposes` stored and read back; the question's *This chat has read* and *You did not
  write this address* (or search words); the search key in the credential store, with `search.json` moved in once; section
  8's encryption design, proposed. Checked on Windows: the gate and session tests (138 with the grant's and the file
  tools'), the mark through a compaction and a change of model on Gemma 4 12B, the session's 40 (the key kept, replaced
  and removed in Credential Manager), the page's 106, every target built in `x64-claude-verify`. *Finished 2026-11-19
  (`0.21.0-dev+157`):* the macOS and Linux key stores built, which found two defects Windows could not (the keychain
  wrapper had no move, and libc++ has no `std::chrono::parse`, so a stored time is read by hand), and a third in the
  build: libsecret brings `-pthread`, which Clang refuses against a `std` module built without it, so on Linux the
  module and every importer are built with it (`cmake/StdModule.cmake`). The mark's tests with their checks removed one
  by one: 22 mutants, each failed by a test but one that cannot differ from the code (an allowance checked after it
  is withdrawn); three survivors first, so the gate's read test now reaches files, the file tools' test checks where
  each result comes from, and the agent's tool events are checked to say whether a call exposed the chat. Mila Chat's
  console says what a chat has read, and offers *for this conversation* only where the gate honours it. Mock scenario
  15, a page that tries to direct Mila. Checked: Windows, the gate, session, file and agent tests (170, the agent's
  on Gemma 4 12B), the session's 40, the page's 107, every target built; Linux (WSL, CPU-only), the library's 116 and
  the session's 40, the key kept, replaced and removed in a Secret Service keyring; macOS, every target built, the
  library's 117, the session's 39, the page's 106; scenario 15 in the window over the mock at 1280x800 and 900 px in
  light and dark; the real session over stdio on Gemma 4 12B FP4 in a scratch store: a chat asked to read and change
  `notes.txt` asked for a folder, read the file and asked to edit it with *Allow once* and *Deny* only, saying what it
  had read; the stored chat kept the three sources and each call's `exposes`; reopened in a session started again, its
  next edit was asked the same way. **Not done:** the macOS key store in the login keychain, which SSH cannot reach
  (`~/mila-work/keychain-test.command`, run once at the Mac); the window on the real session; Chat's prompt run live
  (built on Windows and macOS); a note waiting for the person (5.6) has nothing to wait on until phase 7. A failed call
  still exposes a chat (filed).
- **Phase 4. Projects** (the vision's step 8). The store's `Projects/`, its records and its events; the sidebar's
  projects; *All projects* (4.4); a project's head, glance and chats (4.5); *New project*; moving a chat by its menu; the
  overview's projects; a project's folders as sources (section 10); deletion as section 8 says. Built in this order:
  (a) `Desktop.SessionProcess` and `Desktop.ConversationStore` keep their bodies in `Implementation/`, by CLAUDE.md's
  rule for a plain class, moved unchanged and checked before anything else changes; (b) the workspace folder (section
  8); (c) the project store, `history.jsonl` and `activity.jsonl`, and section 10's messages; (d) the page; (e) mock
  scenarios, then the real session. *Built 2026-11-19 (`0.21.0-dev+158`)*, as sections 4.1, 4.5, 8 and 10 record: the
  store layer is `Desktop.StoreFile` (whole-file writes and records a line), `Desktop.WorkspaceFolder`,
  `Desktop.ConversationStore`, `Desktop.ProjectStore` and `Desktop.EventLog`, in `Apps/Desktop/Session/Src/Store/`; the
  page's projects are `views/projects.js` and `views/new-project.js` over `state.js`; mock scenario 16 is a workspace
  with three projects. Checked on Windows: the session's tests (55, the store's 15 new), the page's 122 and the window's
  9, every target built in `x64-claude-verify`; the session over stdio with no model (a project made, a chat moved in, a
  folder given and held by a new chat at the project's access, narrowed and taken back, the project deleted keeping its
  chats, a deleted chat forgotten in the history); scenario 16 in the window over the mock at 1280x800, 900 and 860 px in
  light and dark, with no console error, nothing scrolling sideways and no network request; and the real session over
  stdio on Gemma 4 12B FP4 in a scratch workspace: a chat begun in a project of one folder read `notes.txt` with no
  folder question and its first write asked to change files there; after the project was set to *Read and change*, the
  next chat's write asked without that step; both chats stored with the project. Linux (WSL, CPU-only, where the session
  process is not built without a GPU backend): the session's tests and the page's. macOS: the session and its tests
  built and passing, the page's tests, and the same stdio check against the Mac's session, whose default workspace is
  now `~/Library/Application Support/Mila`. The key store's test skipped on both, which needs a keyring the run could
  not unlock (no display in WSL, SSH on the Mac). **Not done:** the window on the real session; Q4_0 on the PC (it has FP4 only); the prototype.
- **Phase 8. Made and where it came from.** The creations index, digests, the *Made* drawer (4.9), *Keep* under a reply.
  First because it uses the records and asks nothing new of a model. *Built 2026-11-19 (`0.21.0-dev+159`)*, as sections
  4.3, 4.5, 4.8, 4.9, 8 and 10 record: `Desktop.Creations` reads a chat's creations from its calls and compares a file
  with the disk; the page's drawer is `views/made.js`; mock scenario 17 is scenario 16's workspace with files written in
  a plan's step, one changed since, and a reply kept. Checked on Windows: the session's tests (64, the 9 new ones
  `Desktop.Creations`'), the page's 130, the session and its tests built in `x64-claude-verify` (the window's code is
  unchanged); scenario 17 in the window over the mock at 1280x800, 900 and 860 px in light and dark, with every check
  passing, nothing scrolling sideways and no network request; and the real session over stdio on Gemma 4 12B FP4 in a scratch workspace: a chat
  in a project read `notes.txt` and wrote `summary.md`; the list said it was the same on disk at its write's time; its
  drawer named the model, the two reads before it and the file's text as markdown, and the stored digest is the SHA-256
  of the bytes written; after the file was changed by hand the list and drawer said *changed*; the reply was kept and
  opened as a reply. The first such run ended with no file, the model sending `path:="."` and calls not understood, as
  it did in phase 4's run; the second wrote it. macOS: the session and its tests built, 63 passing and the key store's
  skipped as before, the page's 130, and the same stdio run against the Mac's session on Gemma 4 12B **Q4_0**, with
  the same results. Linux (WSL, CPU-only, where the session process is not built without a GPU backend): the session's
  tests, 63 passing and the key store's skipped (no display to unlock a keyring), and the page's 130. **Not done:** the window on the real session; a write during a
  plan step on a real model (the step is checked by the tests only); setting aside a change on disk the person made
  (section 11, question I).
- **Phase 5. Going somewhere, and finding things by asking** (the vision's step 9). `open_place` first, then
  `find_records` over past chats, `turn.went` (5.2); measured before on by default. This is where a person first feels
  the vision: *find the invoice from six months ago*. A folder is searched by `find_files` and `search_text` (phase 5's
  decisions). *Built 2026-11-20 (`0.21.0-dev+160`)*, as sections 5.2 and 10 record, with `ToolResult` in `Mila::Agents`
  so a result names its own origin (`../ToolCalling.md` 3.3). Checked on Windows: the gate's tests with two new ones
  (a result in the program's words exposing nothing from a tool declared Outside, and one from outside exposing from a
  tool declared the program's own), the file, page-reader and grant tests, 137 with one skip for case; the agent and
  session tests on Gemma 4 12B, 99 with Llama's two skipped (its install predates `architecture_version`); every target
  built in `x64-claude-verify`; the session's 85, 21 new; the page's 139, 9 new; mock scenario 18 in the window over the mock at 1280x800, 900 and 860 px in light and
  dark, with no console error, nothing scrolling sideways and no network request, which found the bar beginning a new
  chat in the project Mila had just opened (fixed: 5.2); and the real session over stdio on Gemma 4 12B FP4 in a seeded
  workspace: *Show me all my projects* opened them and answered from the list; *Open the garden project* opened it; two
  questions about earlier chats were found and cited; *my passport number* was said not to be found; in a chat that had
  read a page, *Open the calculator* asked the person to choose between the project and a chat, with no name reaching
  the model, and `find_records` was asked about, its one match from a chat that had not read from outside exposing
  nothing. Once, after finding the tomatoes, the model also opened that chat unasked; in a later run the same request
  in the exposed chat went to `find_files` instead. macOS: the session, its tests and the library's built; the
  session's 83 with the key store's skipped as before, the gate's 123 with two skipped, the page's 139; and the same
  stdio run on Gemma 4 12B **Q4_0**, every turn right, the exposed chat's list of projects giving only how many, with Q4_0
  sometimes opening *All projects* before the project it was asked for. Linux (WSL, CPU-only): the session's tests, 83
  with the key store's skipped (no display to unlock a keyring), and the page's 139. **Not done:** every model's final
  run; the window on the real session. *Measured 2026-11-20* (5.2): talk-only questions 30 of 32 with no call, going somewhere 14 of 18, finding 8
  of 10, nothing to find 6 of 6, a choice between two places 0 of 2; and after the changes it pointed to (5.5, 5.2's
  two passages, `GrantRefusal`), 30 of 32, 18 of 18, 10 of 10, 6 of 6 and 2 of 2, so both stay in every chat's prompt.
  Checked again after them: the session's 85 on Windows and 84 with the key store's skipped on the Mac and in WSL
  (with the page's 139), the gate's
  with a new test (a refusal by the grant exposes nothing, a tool's own failure still does), the agent tests on Gemma 4
  12B, and every target built.
- **Phase 6. Mila proposes.** `propose_project`, `move_to_project` by question A, the overview's card and the
  answered-in-words offers (5.3, 5.4), undo.
- **Phase 7. What Mila understands.** `note_understanding`, waiting notes, the project's summary and *Next* (4.5), given
  to each chat as section 5.6 says.
- **Phase 9. Plans and runs.** A plan kept as a project's process, the plan drawer, *Run again* (section 7).
- **Phase 10. Findings.** Section 9. Links are not in it.
- **Phase 11. Work while you are elsewhere** (the vision's step 10). Questions 8 and 9, when a library task and more than
  one stream exist; a run's record moves to the task's trace whenever `AgentTasks.md` phase 5 lands.

Calendar, mail and acting on them (the vision's steps 11 and 12) come after these, each a source in section 4.12 with a
design of its own.

## 13. Relationship to other specs

- **`../MilaDesktop.md`** keeps the transcript, the limits, Settings, the session and its schema; its 3.3 and 3.4 are
  replaced at phase 2, its 3.1 at phase 3, and its section 4 gains section 10's messages in the change that builds
  each.
- **`AssistantOnboarding.md`**: the setup ends on the overview, and its section 4.2 is this design's rule on contacting
  the publisher.
- **`../Agents/AgentTasks.md`** is where a run's record goes (section 7).
- **`../Agents/Compaction.md`** must keep a chat's mark (section 5.6).
- **`../ToolCalling.md`** owns the gate every offer and placement passes through, and the grant a project's folder is
  offered under. Section 5.6 is a proposal for its open decision 2, and its 3.3 changes if sections 5.6 and 5.7 are taken.
- **`../Direction.md`** 6.4 is why projects, creations and findings are the app's.
- **`../Direction/MilaVision.md`** is where this design is going: a personal assistant for a person's digital life, of
  which projects are one part. Round 3 holds the unbuilt phases to its sections 3, 6, 7 and 8.
