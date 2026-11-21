# The Mila Vision Discussion

**Role:** Notebook, append-only. **Date:** 2026-11-18, against `0.21.0-dev+153`.

The complete record of the discussion that produced [`../Direction/MilaVision.md`](../Direction/MilaVision.md). Todd
asked that nothing be lost, so sections 1 and 2 keep every message word for word, typing slips included. This notebook
is evidence. The vision spec is what decides, and where the two differ the spec is right.

ChatGPT's text is outside input under advisement. Section 3 sorts it claim by claim.

| Section | What it holds |
|---|---|
| 1 | Todd's chat with ChatGPT, verbatim |
| 2 | The discussion between Todd and Claude, verbatim |
| 3 | Every claim of the ChatGPT chat, sorted, and where it went |
| 4 | What was read in the tree, and what it showed |
| 5 | External sources, and how far each was read |
| 6 | Corrections to what was said in section 2 |
| 7 | What Todd decided, in order |
| 8 | Vision step 1: everything that can leave the machine from Mila Assistant |

---

## 1. The chat with ChatGPT

Todd pasted this into the session, introducing it: "Good morning Claude, In this session I want to discuss a high-level
Mila design chat I had with ChatGPT. Please look at the chat as input to where we could go with Mila."

### 1.1 Todd to ChatGPT

```text
In a chat we had a few days ago we talked about a project based up for an intelligent assistant. The assistant was Mila. I’ve been thinking about the scope of the assistant and although the workspace projects are a good component the scope is not general enough. I see the Mila Assistant as a a safe, secure, private and most importantly a local personal assistant. It helps you navigate your digital life.
```

### 1.2 ChatGPT's response

```text
I think this is an important evolution in how you're thinking about Mila Assistant. And I suspect it actually makes the project-based workspace more meaningful, rather than less.
You're describing something considerably broader than an AI application with a project management interface.
You're describing a personal intelligence environment that helps a person navigate their digital life.
That is a different product philosophy.
1. Reconsidering the purpose of Mila Assistant
I would start with this statement:
Mila is your private, local personal assistant. It helps you understand, organize, navigate and act within your digital life, with your data and your intelligence remaining under your control.
There are four particularly important attributes here:

* Private: Your personal information belongs to you.
* Secure: Your digital life is protected, and actions are performed with appropriate authorization.
* Local: Intelligence and personal data are processed locally wherever possible. The user controls the environment.
* Personal: Mila understands your context, your interests, your ongoing activities and the relationships between them.

I would argue that local is especially important. It isn't simply a deployment option or a performance characteristic. It is part of the product's identity.
2. Digital life is much bigger than projects
Think about the things a person actually needs assistance with.
Workspaces and projects
Ongoing work, research, planning, personal interests, goals, and creations. These have structure, history and artifacts.
Personal information and knowledge
Documents, photographs, notes, files, records, knowledge and information distributed across the user's devices.
Personal organization
Calendar, commitments, reminders, planning, tasks, and keeping track of things that matter.
Discovery and research
Finding information, investigating questions, understanding subjects, comparing alternatives and making decisions.
Communication
Email, conversations, correspondence, preparing responses and keeping track of important interactions.
Activities and actions
Completing tasks, coordinating activities, creating things and helping the user move from intention to outcome.
Notice something important: projects are only one of these areas.
Some activities are persistent and structured. Others are temporary, spontaneous or transactional.
Someone might ask Mila to:

* Find an invoice from six months ago.
* Help plan a holiday.
* Explain something in a document.
* Research a new subject.
* Organize photographs.
* Remember an important commitment.
* Develop a software project.
* Help resolve an administrative problem.

Not all of these should become projects.
But they should all be natural interactions with the same intelligence.
3. The interface should reflect the user's intent, not the underlying AI machinery
This is where I would revisit our previous discussion.
We had arrived at the idea:
The user brings intent. Intelligence develops it into understanding, structure, processes and creations.
I still think that is the right foundation.
I would now take it a step further.
The user shouldn't have to decide whether something is a chat, a project, a research session, an agent workflow or a document creation.
They should simply express what they want to accomplish.
Mila determines the appropriate structure.
Mila Assistant
A PERSONAL INTELLIGENCE ENVIRONMENT
What would you like to do?
Good morning. Where shall we start?
Ask something, find something, continue something, or start something new.
What can I help you with?
Your world
Projects
Your ongoing work
Knowledge
Your personal information
My day
Plans and commitments
Recent activity
Pick up where you left off
Conceptual interface sketch, not a proposed final navigation structure.
The interface is an entry point into a person's digital life, not a catalogue of AI capabilities.
4. The architectural implication is quite significant
I would separate three concepts.
Mila Assistant — the personal environment
The persistent human-facing application. It provides the interface, permissions, personal context, integrations, and continuity.
Mila::AI — the intelligence
The intelligence that interprets intent, reasons, develops plans, uses capabilities, and produces results.
Mila::Agents — the execution layer
The mechanisms that carry out work, coordinate agents, manage processes and report outcomes.
And there is a fourth concept that I think deserves particular attention:
Personal Context — the user's digital world.
This is not simply conversation history or a vector database.
It is the evolving representation of the user's world:

* Information and knowledge.
* Relationships between people, things and events.
* Projects and ongoing activities.
* Decisions and their history.
* Artifacts and their provenance.
* Permissions and access boundaries.

This context is what allows Mila to become genuinely personal without requiring the user to explain everything repeatedly.
It also gives the project-based workspace a natural place: projects become one type of structured activity within the broader personal context.
5. One principle I would make non-negotiable
Mila should never quietly take ownership of the user's digital life.
This connects directly to our earlier discussion about evolving structure.
Mila can discover relationships, suggest organization, identify opportunities, develop plans and propose actions.
But the user remains in control.
I would establish three distinct levels:

1. Understand: Mila can inspect and reason about authorized information.
2. Propose: Mila can suggest changes, plans and actions.
3. Act: Mila can execute authorized actions, with explicit approval where appropriate.

This is particularly important for a product whose identity is privacy, security and trust.
6. The bigger idea
I think there is a much more interesting way to describe Mila now.
A traditional personal assistant is primarily a command interface. You ask it to do something, and it responds.
A traditional productivity application gives you tools to organize your own work.
Mila should be neither.
Mila is an intelligence layer between a person and their digital life.
It helps them understand what is happening, find what they need, organize what matters, explore what interests them and accomplish what they intend.
And it does so in a private, local environment that evolves with them.
That is a much more ambitious vision, but importantly, it is also a coherent one.
One question I think we should explore next
There is a distinction that could fundamentally shape the architecture and interface:
Should Mila primarily be organized around the user's digital world, or around the user's activities within that world?
For example:

* A digital-world-centric Mila starts with information, files, knowledge, people, events and relationships. Activities emerge from those things.
* An activity-centric Mila starts with what the person wants to accomplish. Mila discovers and brings together the relevant information, tools and processes.

My instinct is that the second is the stronger starting point, with the first providing the underlying persistent context.
That would make Mila less like an operating system for files and more like an intelligent companion for navigating life.
```

### 1.3 Todd to ChatGPT

```text
The 2nd choice is quite a nice way of pairing human and digital intelligence. The 1st gives a move structured approach to the engineering on the LLM tech stack / platform
```

### 1.4 ChatGPT's response

```text
I think you've identified something quite important. These two perspectives aren't competing architectures. They are two sides of the same system, and they serve different purposes.

In fact, I would preserve both deliberately.

**The human interacts through activities. The intelligence operates over a model of the digital world.**

That gives us a very interesting separation.

## **Two perspectives, one Mila**

Human perspective

# **Activities and intent**

The user expresses what they want to accomplish.

Find something

Understand something

Develop a project

Organize something

Create something

Get something done

Intelligence boundary

**Mila::AI**

Interpret · Reason · Plan · Coordinate · Learn

Digital perspective

# **Personal digital world**

The underlying information and resources on which intelligence operates.

Information

Relationships

Knowledge

People

Events

Artifacts

Projects

Permissions

## **Why this is architecturally powerful**

It gives you two independent ways of thinking about Mila.

**1. The human-facing architecture**

This is where the experience evolves.

A person might begin with a vague intention:

"I want to understand whether I should replace my car."

That could evolve into:

- Researching alternatives.
- Comparing costs.
- Reviewing existing vehicle information.
- Finding local dealerships.
- Creating a financial comparison.
- Producing a final decision document.

It might eventually become a persistent project, but it doesn't need to start as one.

The structure emerges from the activity.

**2. The intelligence-facing architecture**

This is where you can engineer more formally.

The intelligence needs to understand and operate over:

- Context and memory.
- Resources and their capabilities.
- Relationships and provenance.
- Task graphs and execution state.
- Evidence and confidence.
- Permissions and security boundaries.
- Artifacts and their lifecycle.

These are much more stable engineering abstractions than the potentially infinite variety of human activities.

This also fits your original Mila::AI ambition: the caller expresses intent, and intelligence develops the appropriate recipe for accomplishing it.

## **An important architectural distinction**

I would resist making the human-facing activity model identical to the underlying execution model.

For example:

| Human concept      | Intelligence/platform concept    |
| ------------------ | -------------------------------- |
| Activity           | Task graph                       |
| Goal               | Intent representation            |
| Research           | Retrieval and evidence           |
| Project            | Persistent context and artifacts |
| Remember           | Context and memory management    |
| Create             | Artifact production              |
| Get something done | Agent execution                  |

These are related, but they aren't interchangeable.

A project might contain several activities. An activity might involve several projects, information sources and agents.

That separation gives Mila room to evolve without forcing the interface to expose the complexity of the underlying platform.

## **My proposed guiding principle**

I'd write this into the Mila Assistant architecture:

> **Mila is organized around human intent and activities, while its intelligence is grounded in a structured representation of the user's digital world.**
>
> The human-facing experience remains natural and flexible. The underlying intelligence platform provides the structure, context, security and execution capabilities required to make that experience reliable.

I particularly like this direction because it gives you a stable foundation for the C++ intelligence platform without constraining what the assistant can eventually become.

And there's one more thought worth keeping in mind:

**The user should not have to organize their digital life before Mila can help them navigate it.**

Mila should progressively discover, understand and organize that world through its interactions with the user.

I think that's potentially one of the defining characteristics of Mila.
```

---

## 2. The discussion between Todd and Claude

### 2.1 Claude's first assessment

```text
I think the wider scope is right, and it's a bigger change than the chat suggests. Mila Assistant would go from an agent that works in folders you hand it to the holder of a person's private life. That makes safety the main design problem, and the chat barely touches it. This is reasoning from the specs, with nothing measured, and I changed no files.

## How the chat's claims sort

**Already decided:** intent first, structure emerging from the work, and not having to choose "chat or project" up front are all in section 1 of [AssistantInterface.md](Specifications/Apps/AssistantInterface.md). Understand, Propose and Act is close to our tool gate and grant. Nothing built is wasted, since a chat already needs no project.

**Out of date:** it separates `Mila::AI` from `Mila::Agents`, but there is only `Mila::Agents` (`Direction.md` decision 2). Its "activities" are also not the cards you removed on 11-16, so that word needs care.

**New and worth taking:**
- Digital life as the scope.
- Your pairing: people work through activities, and the engineering works over a structured model of their world.
- Personal Context as a named concept.

**Needs tightening:**
- "Local wherever possible" is too weak a promise. Mail and calendars live on other people's servers, so the promise has to say exactly what stays on the machine.
- "Progressively discover your world" conflicts with today's design, where a chat starts with no folder and asks when it needs one. Discovery needs standing read access, and that is a real decision.

## What the chat misses

**Local doesn't stop prompt injection.** Once Mila reads mail, anyone in the world can put text in front of it, next to private data and tools that can act. Today, every tool call except `update_step` asks the person, and that is what keeps this safe. An assistant that handles your mail can't ask about everything, and people stop reading prompts they see all day. `ToolCalling.md` still lists "rules after untrusted content" as open decision 2. For this product it becomes the central one.

**This is where your structured side pays off.** If every item in Personal Context records where it came from, who wrote it and what it may be used for, the gate can enforce that. Text from a stranger's email could then inform an answer but never choose an action or send private data out.

CaMeL ("Defeating Prompt Injections by Design", arXiv 2503.18813) works along these lines. Going by the abstract only, it solved 77% of AgentDojo tasks with provable security, against 84% undefended. Mila has an unusual advantage here, because `ToolGate` is already the one path every call takes.

**The index becomes the most sensitive file on the machine.** The spec describes conversations as one JSON file each and mentions no encryption, and our threat model doesn't cover other programs running as the same user. "Secure" will need encryption at rest, with keys held by the operating system.

**A wrong memory costs more here.** A misremembered commitment is worse than a bad chat answer, and ContextProfile has already seen confident wrong recalls. Mila should answer from a cited source, not from what the model recalls.

**Some foundations don't exist yet:**
- Work while the window is closed, which is out of the design until phase 11.
- More than one stream at a time, so indexing doesn't compete with the conversation.
- Any retrieval or embedding design.
- Sign-in tokens the model can't reach.
- Connectors that work equally on Windows, macOS and Linux.

## Where I would take it

1. **Keep Personal Context as an index, not a second copy of someone's life.** Originals stay where they are. Everything Mila derives points back to its source and can be deleted and rebuilt. That also delivers "never quietly take ownership". I would not start with a grand graph of people and relationships, because a local model's extraction errors would become stored "facts".
2. **Apply Understand, Propose and Act per source, as a ladder over releases.** Start read-only with local files and past chats: "find the invoice from six months ago". That needs no accounts, works on all three platforms and is easy to measure. Then calendar, then mail, with acting last.
3. **Put the labels and their enforcement in `Mila::Agents`, and connectors and the store in the app.** The gate has to read the labels, and a developer should be able to build the same thing. This changes the current line that retrieval stores are purely the app's concern (`Direction.md` 6.4).
4. **Keep it out of 0.21.** The one thing worth doing sooner is checking that the phase 4 store can carry source information, since it isn't built yet.

## Your decisions

- **Purpose:** does this replace "showcase for people who run local models" (`Direction.md` 1.3)? It widens the audience to ordinary people, and "Your Workspace" may then be too small a name.
- **The promise:** what exactly does "local" guarantee?
- **The boundary:** do you agree with point 3?

I would settle the first one first, because the other two depend on it. Do you see Mila Assistant as becoming that product, or as the direction it grows toward while staying the showcase for now?
```

### 2.2 Todd

```text
I do see the Mila Assistant as the top level product. However, the underlying Mila Library, the LLM tech stack/platform underpins it all and is equally important.
The idea has always been to build and understand the tech from bottom to top and then use it with a general purpose meta app.
```

### 2.3 Claude

```text
That makes the structure clear: **the technology is built upward, and the Assistant's needs come back down as requirements.** The library is then more than a foundation the app happens to use. I've saved this as your decision. `Direction.md` 1.3 still calls the Assistant a "showcase", which no longer fits, and I'd change that only when you say so.

"General purpose meta app" also tells us how to build it. New abilities should arrive as sources and tools under one set of rules, not as features with their own screens. Mail wouldn't be a mail feature. It would be one more source, with its own labels and permissions. That supports putting the labels and their enforcement in `Mila::Agents`, because the app adds no capability of its own.

**What the Assistant would ask of the stack:**
- **Saved sessions:** resume a conversation with no prefill, which is already `Direction.md` 6.4.
- **Several streams at once:** background work has to give way to the person the moment they type.
- **Two agents on one machine:** a reader with no tools could handle untrusted content apart from the agent that acts (6.5).
- **Embedding and retrieval:** built and understood in Mila like everything else. No spec covers this yet.
- **Source labels through the agent loop:** enforced at `ToolGate`.

**Mila may have something unusual to offer here.** Tool results are spliced into the live KV cache, so the agent knows exactly which token positions came from which source. A program talking to a remote server doesn't have that view. With Observability, we could study how much a stranger's text influenced a tool call. That is an untested idea, and it could only ever be a warning signal. The hard guarantee has to stay in the gate.

**The main risk is that you are one developer with two equally important halves.** `Direction.md` section 7 already warns that the top of the stack draws the attention. I'd use one test to keep them balanced: anything the Assistant gains in the library must be usable from a developer's own program, and shown by a test or sample.

The next decision is what "local" promises. I'd suggest: **your data is read, indexed and reasoned over only on your machine, and nothing leaves it unless you approve that specific thing.** Mail would still arrive from your provider, but what Mila learns from it would stay with you. A frontier model could then only ever be an explicit opt-in. Is that the promise you want Mila Assistant to make?
```

### 2.4 Todd

```text
Yes that is my current thought
```

### 2.5 Claude

```text
Good. I've saved it as your current thinking. The words "that specific thing" are the demanding part, and they have three consequences for what we've already designed:

- **Reading a web page counts as data leaving.** The address goes out, and an address can carry private text. Page reads can't become a silent "allowed by rule" once Mila holds personal data.
- **"Allow for this conversation" can't cover outward actions.** The session can remember tools a person allowed for a conversation. That suits local actions, but it would approve future sends nobody has seen.
- **The approval has to show what is leaving.** That means the exact address, search words or message, and where it is going. "Mila wants to use the web" is not enough.

The promise can also be enforced, not just stated. Tool connections already go through the grant's own transport, which checks each one (`ToolCalling.md` decision 3). That gives us one place to test the rule: with no approval, nothing derived from your data goes out. Connections the app makes outside the tools would need the same check.

There is a limit we should be honest about. Mila can show you exactly what will leave, but a long message or address could hide something a person won't notice. So approval alone isn't enough, and the source labels still have to stop untrusted content from steering what gets sent.

I'd take that up next: which sources Mila trusts, and what content from each one is allowed to cause. Your own words are clearly trusted and a stranger's email clearly isn't, but your own files, downloaded PDFs and mail from people you know sit in between. How would you want Mila to treat those?
```

### 2.6 Todd

```text
I can't give you a detailed answer in a few sentences typed on my iPad screen Claude. We will need discussion and specs as we move forward. Let's use this session as our ultimate direction but take small engineering steps to get there
```

### 2.7 Claude

```text
Agreed. This session is now the destination, and it's saved in memory so later sessions start from it. We'll work out the trust questions properly in discussion and specs, and the 0.21 work carries on as planned.

For a small first step, I suggest a read-only check of **everything that can leave the machine from Mila Assistant today**: tool calls, web reads and searches, model installs, and anything the window or session sends. For each one I'd report what goes out, where it goes, and whether the person approves it first.

It changes no code and needs no design decision from you. It also shows how far today's app is from the promise, which gives us real gaps to fix one at a time.

Shall I do that next?
```

### 2.8 Todd

```text
I'd like to generate a Mila vision spec and notebook from the discussion in this session. I don't want anything lost. Everything we do from now on must build towards this
```

---

## 3. The ChatGPT chat, claim by claim

**Standing** is what the claim is now: *already Mila's* (decided before this discussion), *taken up by Todd*, *proposed* (carried into the vision as Claude's or the chat's suggestion, undecided), *changed* (kept in altered form), *out of date*, or *open*.

| # | Claim | Standing | Where it went, and why |
|---|---|---|---|
| 1 | The scope is a person's digital life, wider than projects | Taken up by Todd | Vision 1. It was Todd's own statement to ChatGPT |
| 2 | Mila is private, secure, local and personal | Taken up, sharpened | Vision 3 turns each word into something checkable |
| 3 | Local is part of the product's identity | Taken up by Todd | "most importantly a local personal assistant" |
| 4 | "Processed locally wherever possible" | Changed | Too weak. Replaced by the local promise, vision 3 |
| 5 | "Actions are performed with appropriate authorization" | Changed | Authorization alone does not answer prompt injection, vision 8 |
| 6 | Six areas of digital life | Proposed | Vision 5, as a map and not a commitment |
| 7 | Not every request should become a project | Already Mila's | `AssistantInterface.md` section 1, "a chat needs neither" |
| 8 | The person brings intent; intelligence develops structure | Already Mila's | `AssistantInterface.md` section 1, decided 2026-11-16 |
| 9 | The person need not choose chat, project or research | Already Mila's | Same; no activity cards |
| 10 | "Mila determines the appropriate structure" | Changed | Mila suggests, and the person decides (`AssistantInterface.md` rule 3, question A open) |
| 11 | The interface sketch: *Your world*, *Projects*, *Knowledge*, *My day*, *Recent activity* | Open | Its own text says it is not a proposed navigation. Vision 14, question 6, for the words. Not taken as a layout |
| 12 | "A personal intelligence environment" as a description | Open | Candidate wording, vision 14, question 6 |
| 13 | Three layers: Assistant, `Mila::AI`, `Mila::Agents` | Out of date | One agents API, `Mila::Agents` (`Direction.md` decision 2) |
| 14 | Personal Context as a fourth concept | Proposed, changed | Vision 6: narrowed to an index with source records |
| 15 | Personal Context includes relationships between people, things and events | Changed | Derived later, with evidence and disposable, vision 6 |
| 16 | Projects are one kind of structured activity inside it | Proposed | Vision 1 and 4; compatible with today's design |
| 17 | Mila never quietly takes ownership of a digital life | Proposed | Vision 6 and 7; made structural by the index design |
| 18 | Understand, Propose, Act | Proposed, changed | Vision 7: per source and per kind of action, and also the order of the work |
| 19 | Mila is an intelligence layer between a person and their digital life | Open | Candidate wording, vision 14, question 6 |
| 20 | Activity-centric on top, digital world beneath | Taken up by Todd | Vision 4; Todd's pairing in 1.3 above |
| 21 | The human model and the execution model stay separate | Proposed | Vision 4 |
| 22 | The table of human and platform concepts | Proposed, changed | Vision 4 restates it in Mila's terms and says what exists |
| 23 | Engineering abstractions are "much more stable" than human activities | Unproven | Plausible and not shown. Not stated as fact in the vision |
| 24 | Structure emerges from the activity (the car example) | Already Mila's | `AssistantInterface.md` section 1 |
| 25 | "Evidence and confidence" as something intelligence operates over | Proposed, changed | Vision 6: answers cite sources. Model confidence is not relied on, given ContextProfile |
| 26 | The user need not organize their life first | Proposed | Vision 5, with its cost: read access |
| 27 | Mila progressively discovers and organizes the world | Open | Conflicts with no folder until asked. Vision 14, question 3 |
| 28 | The "C++ intelligence platform" | Changed | Mila is a library (`Direction.md` section 10) |

**What the chat did not address:** prompt injection, the difference between trust and privacy, protection at rest, the
model's reliability, the need for background work and retrieval, three-platform parity, and the library's equal standing.

---

## 4. What was read in the tree

At `0.21.0-dev+153`, 2026-11-18. Read, not run.

**The tool gate.** `Lib/Src/Agents/Tools/ToolGate.ixx`, `decide`: deny and allow rules decide first; a call with no such
rule runs unasked if its tool is in `allowed_for_conversation_`; otherwise the application is asked. `ForConversation`
adds the tool by name, so the answer covers every later call of that tool, whatever its arguments.

**The Assistant's rules.** `Apps/Assistant/Session/Src/Assistant.SessionProcess.ixx`, `useChat`: the only rule added allows
`update_step`. `askAboutCall`:

- `request_folder` in a chat that already has a folder returns *once*, unasked.
- A file or program call in a chat with no folder returns *once*, unasked, and the grant then refuses it.
- `every_time` is true only for `run_command` and for `request_folder`. Every other tool is offered `allow_once`,
  `allow_conversation` and `deny`.

**A defect found.** `web_search` is therefore offered *allow for this conversation*, after which later searches run
unasked. `MilaAssistant.md` 3.7 says "every search is asked about with its words shown, since they leave the machine", and
`AssistantInterface.md` 5.5 says "a search is asked every time". Filed in `Issues/Untriaged.md`. No test names it. Found
by reading and not reproduced by running the app.

**The tools offered.** `Apps/Assistant/Prompts/general.json`: `propose_plan`, `update_step`, `read_web_page`, `web_search`,
`request_folder`, `list_folder`, `read_file`, `find_files`, `search_text`, `write_file`, `edit_file`, `run_command`.

**The network.** `Lib/Src/Agents/Grants/ToolGrant.ixx`: a grant with the network holds a checked transport that accepts
public addresses. It checks where a connection goes and has no view of what is being sent.

**Other outward paths seen, not a complete list.** `Apps/Assistant/Window/src/main.js`, `openLink`: an allowed link goes to
the person's browser through `shell.openExternal`. `Assistant.WebSearch.ixx` sends the query and key to Tavily. Model
installs reach the hub. `run_command` has the network only when a call asks. Vision step 1 is the full inventory.

**At rest.** `MilaAssistant.md` 3.7: the search key is plain text in `search.json` until the credential store holds it.
A search of `Apps/Assistant` for `safeStorage`, DPAPI, `CryptProtect`, Keychain, libsecret and "encrypt" found only the
Mac signing configuration and packaging, so conversations are not encrypted.

**Confinement.** `Issues/Untriaged.md`: nothing at the operating system's level confines `mila-session`.

**Threat model on record.** `Notebooks/LocalToolService.md` 6.1: the broker design defended against a model steered by
what it reads and against a bad tool server, and not against another program running as the same user.

**Model reliability.** `ContextProfile.md`: Gemma 4 12B Q4_0 with thinking on recalled 78 of 80 planted records at 128K,
and both misses were confident wrong values after about 97 thinking tokens. On the 26B-A4B at 128K, 30 of 35 misses were
confident wrong values.

**Retrieval.** No specification designs retrieval or an embedding model. `Direction.md` 6.4 says "retrieval stores are an
application's concern".

**Today's Assistant design.** `Apps/AssistantInterface.md`: nothing outlives a conversation (section 2), one conversation
at a time (section 2), work with the window closed out of the design until phase 11 (section 11, question 9), phases 1 to
3 built and phase 4, the project store, not.

**Direction.** `Direction.md`: section 1, the trait and its one question; 1.3, the two audiences and "showcase"; 3.3, no
application-private capability; 6.4 saved sessions; 6.5 two agents; section 7 the risk of attention moving to the top;
section 9, no remote model behind `Mila::Agents`.

---

## 5. External sources

| Source | Read | What it was used for |
|---|---|---|
| "Defeating Prompt Injections by Design", arXiv 2503.18813 (CaMeL) | Abstract only, 2026-11-18 | Control and data flow derived from the trusted request, so untrusted data cannot change program flow; capabilities against unauthorized data movement; AgentDojo, 77% with provable security against 84% undefended |
| "Design Patterns for Securing LLM Agents against Prompt Injections", arXiv 2506.08837 | Abstract only, 2026-11-18 | Patterns with provable resistance and a discussion of their trade-offs. The abstract does not state the trade-offs' size |

Neither paper's body has been read, and each should be read before a design depends on it.

---

## 6. Corrections to what was said in section 2

Found while checking the discussion against the tree. Section 2 stays as it was said.

1. **"Every tool call except `update_step` asks the person"** (2.1) was too strong. Three kinds of call go unasked: a
   call of a tool the person allowed for the conversation; `request_folder` when the chat already has a folder; and a
   file or program call in a chat with no folder, which the grant then refuses. The conclusion stands, and is stronger:
   asking is the main protection today, and it can already be turned off per tool for a conversation, web tools
   included.
2. **"The grant's own transport, which checks each one"** (2.5) is true of the address. It accepts public addresses and
   does not know what a request carries. Holding the local promise there needs more than today's check.
3. **"The 0.21 work carries on as planned"** (2.7) was Claude's assumption and Todd has not said it. Todd's later words
   are that everything builds towards the vision. The vision records v0.21.0's scope as unchanged by the document and
   any change to it as Todd's.

---

## 7. What Todd decided, in order

All 2026-11-18.

1. **Mila Assistant is a safe, secure, private and above all local personal assistant that helps a person navigate their
   digital life.** Workspace projects are a good component and not general enough. (1.1)
2. **The activity view pairs human and digital intelligence, and the digital-world view gives the engineering its
   structure.** (1.3)
3. **Mila Assistant is the top-level product. The library is equally important. The idea has always been bottom to top,
   then a general purpose meta app.** (2.2)
4. **The local promise is Todd's current thought:** your data is read, indexed and reasoned over only on your machine, and
   nothing leaves it unless you approve that specific thing. (2.4)
5. **This discussion is the ultimate direction, reached by small engineering steps, with discussion and specs along the
   way.** The trust questions need real discussion. (2.6)
6. **A Mila vision spec and notebook record it, with nothing lost, and everything from now on builds towards it.** (2.8)

---

## 8. Vision step 1: everything that can leave the machine from Mila Assistant

2026-11-18, `0.21.0-dev+153` with the vision's documents in the working tree. Todd approved the step ("Yes"). No product
code was changed.

**How it was done.** Every use of the network was traced by reading `Apps/Assistant/Session/Src`, `Apps/Assistant/Window/src`,
`Apps/Assistant/Ui`, `Lib/Src/Networking`, `Lib/Src/Distribution/Hub` and `Lib/Src/Agents`. The window was also run once on
Windows: Electron 44.9.3 from `Apps/Assistant/Window`, mock scenario 1, no model and no clicks, for 120 s with Chromium's
network log (`--log-net-log`, `IncludeSensitive`). Its user data and temporary files were under `D:\Claude\egress`, where
the log and `read_netlog.py` are kept.

**Not verified.**

- The session was read and not run, and no packets were captured, so section 8.1's session rows are what the code does.
- The network log covers one idle two-minute run of the window on Windows. It does not show what Chromium might do later,
  after interaction, or on macOS and Linux.
- Nothing here covers traffic made by the operating system, the GPU driver, or programs a command starts.

### 8.1 The paths

| # | Path | What goes out | Where | Who starts it | What the person approves | Finding |
|---|---|---|---|---|---|---|
| 1 | `read_web_page` | The whole address, path and query included; `Accept`; `User-Agent: Mila/<version>`; the host name to the resolver | The named host, then up to 5 redirects, to public addresses only. `http` is accepted | The model | Asked, with the address shown. May be allowed for the whole conversation | G2, G3 |
| 2 | `web_search` | The search words, depth and result count; the person's Tavily key as a bearer token; `User-Agent` | `api.tavily.com` over https, POST, no redirects. Offered only with a key | The model | Asked, with the words shown. May be allowed for the whole conversation | G2 (filed earlier) |
| 3 | `run_command` with `network: true` | Whatever the program sends, from the granted folder. Setting off by default | Any public address | The model | Asked every call, showing the command. The question cannot show what the program will send | G6 |
| 4 | A link in Mila's reply | The whole address, plus whatever the browser adds. A `mailto:` address opens the mail program with what it carries | Any host, through the system's browser or mail program | The model writes it, the person clicks | The click. There is no confirmation. The label may differ from the address, which appears only as a hover title | **G1** |
| 5 | Listing published models | `api/models?author=<owner>&full=true`, each manifest, parts of each model's weights; the IP address; `User-Agent`; **a HuggingFace token if one is found on the computer** | `huggingface.co`, and its redirect hosts without the token | The page, as the setup's model step shows, or on opening Models with no list | Not asked | G4 |
| 6 | Installing a model | The model's files and byte ranges, the IP address, `User-Agent`, and the same token | As row 5 | The person presses *Install*, with the model named | The press. The token is not mentioned | G4 |
| 7 | The window's proxy discovery | `wpad` name lookups and WPAD queries on the network adapters | The local network's resolver and DHCP | Chromium, at start and about 8 s later | Not asked | G5 |

The token in rows 5 and 6 comes from `MILA_HF_TOKEN`, `HF_TOKEN` or `~/.cache/huggingface/token`
(`discoverHuggingFaceToken`). `HttpClient` sends a token only to the host a request started on. The window passes its whole
environment to the session.

Rows 5 and 6 use the default transport, not a grant's. It is libcurl on Windows and Linux, where an environment proxy
applies, and Apple's URL loading on macOS, with the Mac's proxy settings, no cache and no cookies. Tool connections use the
grant's checked transport and never a proxy.

### 8.2 The window's network log

In 120 s: 128 events, no URL request, no TCP connection and no bytes reported sent. It held:

- 4 host-resolver requests for `wpad:80`, at start and about 8 s later, with `PAC_FILE_DECIDER` and `WPAD_DHCP_WIN_*` events;
- 2 `UDP_CONNECT` events to `[2001:4860:4860::8888]:443`, beside `HOST_RESOLVER_MANAGER_IPV6_REACHABILITY_CHECK`. The log has
  no `UDP_BYTES_SENT`. A UDP connect by itself sends nothing, so this reads as a check for an IPv6 route.

Whether WPAD runs depends on the computer's proxy setting, so another machine may differ.

### 8.3 Looked for and not found

- **The page:** its policy is `default-src 'self'; img-src 'self' blob:`; an image in a reply is never fetched; it has no
  `fetch` outside the mock's development panel, which is not packaged.
- **The window:** navigation and new windows are refused, spellcheck is off, permissions are limited to writing the
  clipboard, and it starts no auto-updater and no crash reporter. Packaging uses `--publish never`.
- **The session:** no telemetry. Its only network code is `Assistant.WebSearch.ixx`, the model hub through `Assistant.Models.ixx`,
  and the tools given a grant. `ready` lists installed models only, offline.
- **Conversations, settings and the licence record** are written locally and are not sent. The login name in `about` goes
  to the page only.

### 8.4 Outside Mila's connections

These leave through something the person already runs, so Mila cannot see them. They still matter for what the promise
may claim: a file Mila writes into a folder a sync client watches is uploaded by that client, and text copied with *Copy*
reaches a clipboard the operating system may sync.

### 8.5 The gaps, most serious first

| Gap | What is wrong | Filed |
|---|---|---|
| **G1** | A link written by the model can carry private text in its address and goes out on one click, with its address hidden behind any label. Content that steered the model could use this | `Issues/Untriaged.md` |
| **G2** | `read_web_page` and `web_search` can be allowed for a whole conversation. For search this also contradicts today's specs | `Issues/Untriaged.md`, filed earlier |
| **G3** | An approved page read may go on to hosts the person was not shown, and may use unencrypted `http` | `Issues/Untriaged.md` |
| **G4** | Mila contacts HuggingFace without asking when it lists models, and sends a token it found on the computer even for public listings, tying the request to the person's account | `Issues/Untriaged.md` |
| **G5** | The window performs proxy discovery although its page uses no network | `Issues/Untriaged.md` |
| **G6** | A command with the network can send anything to any public address, and approval cannot show what | Existing entry, *A command run through the grant reaches every public host or none*, extended |

**Against the local promise.** No path sends conversation or file content without some action by the person, but rows 1,
2 and 4 do not yet meet "approve that specific thing". Row 3 cannot meet it as designed. Rows 5 to 7 send no conversation
content, and rows 5 and 7 happen without being asked.
