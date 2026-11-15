# Mila Assistant Projects

**Status:** idea, round 1 (2026-11-14). The idea is Todd's, put down so it is not lost; nothing in it is decided.
Rounds with Todd follow in a session of their own. Not admitted to any release.

A project as the place Mila Assistant's work lives: every conversation, of every activity, belongs to one, in
place of today's single list of conversations. Todd, 2026-11-14: "I don't like the clutter of the 'chat' sessions.
I envision a 'Project' structure that captures everything that Mila will be able to do agentically including
chats."

---

## 1. What this would decide

- What a project is: what it holds, and what it hands every conversation inside it.
- How the sidebar shows projects and their conversations, and where *New* starts.
- Which of today's per-conversation choices -- the activity, the folder, the tools' grant -- move to the project.
- How conversations stored before projects find their place.
- What, if anything, a project carries across its conversations, and whether that is the app's or the library's.

It would not decide the activities themselves (`../MilaDesktop.md` 3.1), the model Mila chooses (3.2), or the
first run (`AssistantOnboarding.md`), though it touches the last: a first run may end in a first project.

## 2. Where it stands today

- **The sidebar is one list** (`../MilaDesktop.md` 3.3): every conversation, newest first, grouped by date, each
  row its title and its activity. Nothing groups conversations by what they are for.
- **A conversation carries its activity and, for two of them, a folder** (3.1, 6): Ask & Explore has no tools and
  no grant; Build & Code has the folder read-write; Research & Analyze has the network; Get Things Done has both.
  The folder is chosen when the conversation starts, in the window's own picker (`folder.choose`), and a
  conversation whose folder is gone is not opened.
- **The store is one file per conversation** in the app's folder (`Desktop.ConversationStore.ixx`, section 6):
  its messages, the history the model read, its activity, its folder and its plan. `conversations.list` carries
  `id`, `title`, `updated`, `model`, `activity` and `folder`.
- **Nothing outlives a conversation** except the conversation itself: no instructions, files or memory shared
  between conversations. `Mila::Agents::Session` is what the program gave one agent and its conversation
  (`../Direction.md` 4.3).

## 3. Rules it stands on

- **An app adds no capability to the library** (CLAUDE.md). Grouping and showing conversations is the app's. A
  memory carried across a project's conversations, or a project's files read into a conversation, would be what
  an agent can do, and so `Mila::Agents` work, decided there.
- **The activity decides what a tool may reach** (`../ToolCalling.md` 3.4). A project that owns a folder offers
  it; whether a conversation reads it, writes it or does not see it stays the activity's grant. Being in a
  project never widens a grant.
- **Conversations on disk are the person's data.** Those stored before projects are kept and placed, never
  dropped.
- **End-user prose** (CLAUDE.md): *project* is the person's word, and nothing about one is explained by how it is
  stored.

## 4. What it would touch

Named so the cost is seen before the case is made; none of it is designed.

- **The page:** the sidebar (projects, their conversations, *New* inside a project and outside one), the home
  screen, the header.
- **The session's messages** (`../MilaDesktop.md` 4): `conversations.list`, `conversation.new`,
  `conversation.loaded` and `folder.choose` gain a project, and projects need their own list, create, rename and
  delete.
- **The store:** where a project and its conversations live, and placing the conversations already stored.
- **The grant:** a folder chosen once for many conversations is a wider standing grant than one chosen per
  conversation, and is written down as such before it is built.
- **The library, only if** a project carries anything across its conversations (section 5, question 3).

## 5. Open questions

1. *Is everything in a project?* Or do quick questions live outside any, in a default place the person never
   names?
2. *Does a project own one folder*, and is that then the only place a folder is chosen? Can a project have none,
   or more than one?
3. *What else a project holds:* instructions for every conversation in it, files, a memory of earlier work. Each
   Instructions ride on the system prompt the app already gives (`Session::withSystemPrompt`); files read into a
   conversation and memory across conversations are agent capability (section 3).
4. *The activity inside a project:* does each conversation keep its own, as today, or does the project set one?
5. *The first run:* does it create a first project, and named what?
6. *Placing the conversations stored before projects:* one project for them all, a project per folder, or the
   default place of question 1.
7. *Deleting a project:* what happens to its conversations, and to its folder (never touched, presumably).
8. *The release:* 0.21.0 is dated 2026-12-04. This would be admitted to it only by Todd, against a ROADMAP
   criterion or as a scope decision, else it goes to `Issues/Vnext.md`.
