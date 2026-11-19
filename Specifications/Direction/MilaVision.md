# The Mila Vision

**Role:** Direction. **Written:** 2026-11-18, against `0.21.0-dev+153`, from a discussion between Todd and Claude that
began with a design chat Todd had with ChatGPT. The whole discussion, word for word, and the evidence read for it are
in [`../Notebooks/MilaVisionDiscussion.md`](../Notebooks/MilaVisionDiscussion.md).

**This is where Mila is going. Everything we do from now on builds towards it** (Todd, 2026-11-18). We get there by
small engineering steps, each discussed and specified (section 11).

Every statement here carries one of four standings, because a vision that blurs them turns a suggestion into a
commitment:

| Standing | Meaning |
|---|---|
| **Decided** | Todd said so. Changing it is Todd's. |
| **Current thinking** | Todd agreed with it as where things stand today. Design on it, and expect it may move. |
| **Proposed** | Claude's recommendation. Nothing rests on it until Todd takes it up. |
| **Open** | A question that needs discussion. |

Text from ChatGPT is input under advisement. It appears here only where Todd took it up, or as a proposal or question
marked as such.

---

## 1. The vision

**Decided (Todd, 2026-11-18).**

**Mila Assistant is the top-level product:** "a safe, secure, private and most importantly a local personal assistant.
It helps you navigate your digital life."

**The Mila library is equally important:** "the underlying Mila Library, the LLM tech stack/platform underpins it all
and is equally important."

**The two are one project built in one direction:** "The idea has always been to build and understand the tech from
bottom to top and then use it with a general purpose meta app."

Workspace projects are "a good component", and not the whole scope. Someone may ask Mila to find an invoice from six
months ago, plan a holiday, explain a document, research a subject, organize photographs, remember a commitment,
develop software or sort out an administrative problem. These are all natural requests to the same intelligence, and
most of them should never have to become a project.

## 2. One stack, two equal halves

```
             the person: what they want to understand, find, organize, make or get done
                                           |
Mila Assistant        the general purpose meta app: one conversation, every ability, the person in control
                                           |
Mila::Agents          the agent, its session, tools, the gate, the grant, tasks
                                           |
Mila library          model handle, deployment planning, models, components, operations, tensors
                                           |
Backends              CUDA, Metal, CPU host
```

**The technology is built upward, and the Assistant's needs come back down as requirements.** That is how the two
halves stay equal: the library is not a finished foundation under an app, and the app is not a demonstration on top
of a library.

**A general purpose meta app gains abilities as sources and tools under one set of rules, not as features with
screens of their own** (*proposed*, as what "meta app" asks of the design). Mail would not be a mail feature. It would
be one more source with its labels and permissions, and tools that act on it through the same gate. This continues
the decision that there are no activity cards (`../Apps/AssistantInterface.md` section 1).

Three rules hold the halves together:

1. **An app adds no capability to the library; it exposes what the library has** (`CLAUDE.md`; `../Direction.md` 3.3).
   This stays in force and matters more now. An app's own tools and connectors remain its concern
   (`../ToolCalling.md` section 5).
2. **What the Assistant gains in the library is usable from a developer's own program,** and a test or sample shows
   it (*proposed*).
3. **The library keeps its own test.** `../Direction.md` section 1's question, whether a release lets a developer
   harness intelligence more directly, stays the library's. The Assistant adds a second test, it does not replace that
   one.

## 3. The promises

Each word in the vision has to mean something that can be checked. A promise nobody can test is marketing.

| Word | What it promises | Standing |
|---|---|---|
| **Local** | **Your data is read, indexed and reasoned over only on your machine, and nothing leaves it unless you approve that specific thing.** | Current thinking (Todd, 2026-11-18) |
| **Private** | What Mila holds about you is yours: you can see it, trace it and delete it, and it is protected at rest. | Proposed |
| **Secure** | Content Mila reads cannot make it act against you, and each action runs under what you allowed. | Proposed |
| **Safe** | Mila does not quietly take over. It understands, proposes and acts at levels you control, and what it did can be seen and, where possible, undone. | Proposed |
| **Personal** | Mila knows your context, so you do not have to explain your life again each time, and it can show where what it knows came from. | Proposed |

**What the local promise means.** Mail still arrives from your mail provider and a calendar from its server. What Mila
learns from them stays with you. The model, the index and the reasoning are on your machine. A frontier model could
only ever be an explicit opt-in for a specific request, since it would send data out.

**What "that specific thing" requires** (*proposed*, each a consequence of the promise):

- **Reading a web page counts as data leaving.** Its address goes out, and an address can carry private text.
- **An approval "for this conversation" cannot cover an outward action,** because it approves sends nobody has seen.
- **An approval shows what is leaving and where it is going:** the exact address, search words or message, never only
  "Mila wants to use the web".
- **Approval is not sufficient by itself.** A long message or address can hide something a person will not notice, so
  section 8's rules must also stop untrusted content from steering what is sent.

**The promise is tested.** The conformance check is: with no approval, no connection carrying anything from the
person's data is accepted. Tool connections already pass through the grant's own transport (`../ToolCalling.md` decision
3), so that is where it can be held. Connections the app makes outside the tools need the same check.

## 4. Two perspectives, one Mila

**Current thinking (Todd, 2026-11-18):** the activity view is "quite a nice way of pairing human and digital
intelligence", and the digital-world view "gives a [more] structured approach to the engineering on the LLM tech stack
/ platform".

- **The person works through intent and activities.** They say what they want. They do not first decide whether it is
  a chat, a project, research or a document.
- **The engineering works over a structured representation of the person's digital world:** sources, records, people,
  events, what was made, where each came from, and what each may be used for.

**The two are kept separate** (*proposed*). The human model can stay natural and open-ended because the engineering
model below it carries the structure. A project may hold several activities, and one activity may touch several
projects, sources and agents.

| What the person experiences | What the stack works with | Where Mila stands |
|---|---|---|
| Something I want to get done | Intent, then a plan and its steps | `propose_plan` today; the library's task later (`../Agents/AgentTasks.md`) |
| Research | Retrieval, evidence and its sources | Web tools today; retrieval not designed |
| Project | Lasting context and what was made in it | `../Apps/AssistantInterface.md` phase 4 onward |
| Remember | Personal Context and session memory | Section 6; `../Direction.md` 6.4 |
| Make something | A file or reply, and where it came from | *Made*, `../Apps/AssistantInterface.md` rule 4 |
| Get it done | The agent, its tools, the gate and the grant | `Mila::Agents` |

**Words.** *Activity* here means what a person is trying to do. It is not the activity cards removed on 2026-11-16 and
not the *Activity* view of what happened. `Mila::AI` in the source chat is `Mila::Agents`; there is no separate layer
between them (`../Direction.md` decision 2). What ChatGPT calls artifacts are *what Mila made*. Mila remains a library,
never a runtime or engine.

## 5. The scope: a person's digital life

Carried from the discussion as the map of the territory, *proposed* as scope and not a list of commitments:

| Area | Examples |
|---|---|
| **Work and projects** | Ongoing work, research, plans, goals, things being created |
| **Personal information and knowledge** | Documents, photographs, notes, files, records across devices |
| **Personal organization** | Calendar, commitments, reminders, tasks, what must not be forgotten |
| **Discovery and research** | Finding, investigating, understanding, comparing, deciding |
| **Communication** | Mail, conversations, drafting replies, keeping track of who said what |
| **Activities and actions** | Coordinating and completing things, moving from intention to outcome |

**The person should not have to organize their digital life before Mila can help them navigate it** (*proposed*, from
the discussion). That has a cost: discovering a person's world takes read access beyond today's rule that a chat starts
with no folder and asks when it needs one. How far standing access goes is section 14, question 3.

## 6. Personal Context

The evolving representation of the person's world that lets Mila be personal without being told everything again.
Carried from the discussion. The shape below is *proposed*.

**An index, not a second copy of someone's life.** Originals stay where they are. Everything Mila derives points back
to its source and can be deleted and rebuilt. This is how "Mila never quietly takes ownership of your digital life"
becomes a property of the design.

**Each record says:**

- what it is, and a pointer to the original;
- which source it came from and when;
- who wrote it, as far as that can be known;
- whether it may direct an action, and whether it may leave the machine (section 8);
- what it was derived from, if it was derived.

**Answers come from sources.** Mila answers from a cited record, not from what a model recalls. ContextProfile measured
why: on the Gemma 4 26B-A4B at 128K, 30 of 35 recall misses were confident wrong values, and thinking did not recover
Gemma 4 12B's misses at that depth (`../ContextProfile.md`). A misremembered commitment costs more than a poor chat
answer.

**It starts with records that already exist.** Files, past chats, messages and calendar entries come first. A grand
graph of people and relationships does not, because a local model's extraction errors would become stored "facts".
Relationships may later be derived, always pointing at their evidence, and always disposable.

**It is protected.** An index of a person's life is the most sensitive file on their machine. It is encrypted at rest,
with its key held by the operating system's credential store, on Windows, macOS and Linux. Deleting a source removes
what was derived from it.

**Where it lives is open** (section 14, question 4). `../Direction.md` 6.4 says retrieval stores are an application's
concern. The proposal is that the **labels and their enforcement belong in `Mila::Agents`**, because the gate has to
read them and a developer should be able to build the same thing, while **connectors and the store belong to the app**.

## 7. Understand, Propose, Act

Carried from the discussion, with one change: the levels are set **per source and per kind of action**, not once for
all of Mila (*proposed*).

1. **Understand.** Mila may inspect and reason about what the person has authorized.
2. **Propose.** Mila may suggest changes, plans and actions.
3. **Act.** Mila may carry out what is authorized, with explicit approval where appropriate.

The levels are also the order of the work. A source starts read-only. Acting on it comes last, and only once reading
it has been shown to be safe and useful.

**Mila can discover and suggest. The person remains in control** (*proposed*, consistent with
`../Apps/AssistantInterface.md` rule 3, whose question A is still open).

## 8. The central problem: what content may cause

**Running locally does not stop prompt injection.** Once Mila reads mail, anyone in the world can put text in front of
it, beside private data and tools that can act. Local inference removes a cloud provider from the picture. It does not
remove the stranger who wrote the message. This is the hardest problem in the vision, and `../ToolCalling.md` open
decision 2, *Rules after untrusted content*, becomes its centre.

**Asking about every action does not scale.** It is what protects most calls today, but an assistant that handles mail
cannot ask about everything, and people stop reading prompts they see all day.

**Two separate questions about every piece of content** (*proposed*):

- **May it direct an action?** This is about trust in who wrote it.
- **May it leave the machine?** This is about how private it is.

They differ: a public web page may leave freely and must not direct anything; a stranger's email is both untrusted and
private; the person's typed words direct Mila and may still be private.

| Source | May it direct an action? | May it leave? | Standing |
|---|---|---|---|
| The person's own words, in the conversation | Yes | Only with approval | Proposed |
| The app's own prompts | Yes | Not applicable | Proposed |
| A public web page | No | Its address, only with approval | Proposed |
| A stranger's message | No | Only with approval | Proposed |
| The person's own files | No, to begin with, and the cost is counted before this is final | Only with approval | Decided to start (Todd, 2026-11-18, `../Apps/AssistantInterface.md` question E): a file in their folder may have come from anywhere |
| A downloaded document | **Open**, starting from no | Only with approval | Needs discussion |
| Mail from people they know | **Open** | Only with approval | Needs discussion: an account can be compromised |
| What the model wrote after reading any of these | Takes the least trust of what it read | Takes the most privacy of what it read | Proposed |

The last row is the difficult one. Labelling the input is not enough, because a model that has read a hostile message
may carry its influence into everything it writes next. The same applies to a compaction summary, which merges sources
(`../Agents/Compaction.md`).

**Proposed invariants,** each to be made testable:

1. Only the person's words and the app's prompts choose what Mila sets out to do. Other content can inform the work and
   cannot change its course.
2. Nothing from the person's data leaves without approval of that content going to that destination.
3. A label is never lost: not through a tool result, a summary, a derived record or a change of model.
4. Enforcement is in the gate and the grant, in ordinary code. It never depends on the model behaving.
5. Every action can be traced to the request that authorized it and the content that informed it.

**The published idea closest to this** is CaMeL, "Defeating Prompt Injections by Design" (arXiv 2503.18813), which
derives control and data flow from the trusted request so retrieved data cannot change the program's flow, and uses
capabilities to stop private data moving along unauthorized paths. Its abstract reports 77% of AgentDojo tasks solved
with provable security against 84% undefended. Only the abstract has been read, and nothing here has been measured on
Mila. Restrictions of this kind cost usefulness, and how much on local models is unknown. "Design Patterns for Securing
LLM Agents against Prompt Injections" (arXiv 2506.08837) surveys such patterns and their trade-offs (abstract only).

**Where Mila has an advantage** (*proposed*, untested):

- **One path.** `ToolGate` is already the one path a model's call takes to a tool.
- **A reader with no tools.** A second agent could read untrusted content and return constrained results to the agent
  that acts (`../Direction.md` 6.5).
- **Token-level provenance.** A tool result is spliced into the live KV cache, so the agent knows exactly which token
  positions came from which source. A program talking to a remote server does not have that view. With Observability we
  could study how much a stranger's text influenced a tool call. This could only ever be a warning signal. The guarantee
  stays with invariant 4.

## 9. What the Assistant asks of the stack

*Proposed.* Each is a requirement coming down the stack (section 2), and each needs its own design.

| Need | Why | Level | Today |
|---|---|---|---|
| Source labels through the agent loop | Section 8 | `Mila::Agents` | None |
| Rules after untrusted content | Section 8 | `Mila::Agents` | `../ToolCalling.md` open decision 2 |
| Labels that survive compaction | A summary merges sources | `Mila::Agents` | `../Agents/Compaction.md` has no such concept |
| Saved sessions | Resume with no prefill | `Mila::Agents`, models | `../Direction.md` 6.4, after v0.21.0 |
| Several streams over one model | Background work yields the moment the person types | Models, `Mila::Agents` | One conversation at a time (`../Apps/AssistantInterface.md` section 2) |
| Two agents on one machine | A tool-less reader beside the agent that acts | Deployment, `Mila::Agents` | `../Direction.md` 6.5, not designed |
| Tasks that outlive one response | Multi-step work with a trace | `Mila::Agents` | `../Agents/AgentTasks.md`, agreed, no code |
| Work while the window is closed | Keeping up with a person's world | App, library | Out of the design until `../Apps/AssistantInterface.md` phase 11 |
| Retrieval, and an embedding model if measurement calls for one | Finding things in Personal Context | Library, built and understood like everything else | No design |
| Connectors | Mail, calendars and other sources, equal on all three platforms | App, perhaps through MCP | MCP is later (`../ToolCalling.md` section 6) |
| Sign-in tokens the model cannot reach | Connectors hold a person's most valuable secrets | App, operating system | Later (`../ToolCalling.md` section 6) |
| Encryption at rest | Section 6 | App, operating system | None |
| Confinement of the session itself | A defect in the grant should not reach everything the person can | App, operating system | Filed in `../../Issues/Untriaged.md` |

## 10. Where Mila stands today

Read in the tree at `0.21.0-dev+153`. What can leave the machine was inventoried on 2026-11-18
(`../Notebooks/MilaVisionDiscussion.md` section 8); the session was read and not run, and the window was measured on
Windows only.

| Fact | Against the vision |
|---|---|
| Models run locally, and a chat needs no project | In place |
| The gate is the one path to a tool, and each question shows every argument (`../ToolCalling.md` 3.3) | In place, and the base for section 8 |
| Tool network connections go through the grant's checked transport, to public addresses only | A place to hold the promise; it checks the address, not what is being sent |
| Only `update_step` runs by rule, and other calls that can run are asked about | Protects today, does not scale |
| The person can answer *allow for this conversation* for every tool except `run_command` and `request_folder` (`Desktop.SessionProcess.ixx`, `askAboutCall`) | Conflicts with "that specific thing" for `read_web_page` and `web_search` |
| `web_search` can be allowed for a conversation, though `../MilaDesktop.md` 3.7 and `../Apps/AssistantInterface.md` 5.5 say every search is asked | **A defect today,** filed in `../../Issues/Untriaged.md` |
| A link the model wrote opens on one click, with its address visible only on hover | **The most serious gap found:** private text could leave in the address |
| A folder given to a chat can be read and changed at once (`Desktop.SessionProcess.ixx`, `grantChat`) | Short of a source starting read-only (section 7) |
| An approved page read may follow redirects to hosts not shown, and may use `http` | Short of "that specific thing" |
| Listing published models contacts HuggingFace unasked, with any HuggingFace token found on the computer | No conversation data, but unapproved and tied to the person's account |
| The window performs proxy discovery, though its page uses no network | Measured on Windows; no URL request was made |
| A command with the network can send anything to any public address | Its approval cannot show what will be sent |
| The web search key is plain text in `search.json` until the credential store holds it (`../MilaDesktop.md` 3.7) | Must move |
| Conversations are one JSON file each, and `Apps/Desktop` has no encryption code | Must change before Personal Context |
| `mila-session` is confined by nothing at the operating system's level | Filed; matters more as Mila holds more |
| Nothing outlives a conversation but the conversation | Personal Context does not exist yet |
| One conversation runs at a time, and no work runs with the window closed | Foundations are missing |

**Changed since, by `../Apps/AssistantInterface.md` phase T1 (2026-11-18, `0.21.0-dev+155`):** tools that use the
network are asked every time; a link shows its whole address before it opens; a page read does not follow a redirect to
another site or from `https` to `http`; the publisher is contacted only on the person's press and with no token; the
window does no proxy discovery (measured on Windows); a chat's folder starts read-only. What was sent is listed in
Activity. Commands with the network and encryption at rest stand as the table says.

## 11. Small engineering steps

**Decided (Todd, 2026-11-18):** "take small engineering steps to get there", with "discussion and specs as we move
forward".

The order below is *proposed*. Each step is small enough to finish, useful alone, and leaves something measured or
tested. Any change to `Lib/Src` or `Apps/` needs Todd's direction first, and every feature step lands on Windows, macOS
and Linux, with Linux's window stated as unverified because WSL cannot run it (`CLAUDE.md`).

| # | Step | Level | Done when |
|---|---|---|---|
| 1 | **Inventory of everything that can leave the machine.** *Done 2026-11-18:* `../Notebooks/MilaVisionDiscussion.md` section 8; its gaps are filed | Read-only | A table in the notebook gives, for each path, what goes out, where, and who approves it; each gap is filed |
| 2 | **Close what needs no new design:** every search asked, as the specs already say; the publisher contacted only when the person asks, with no token unless a gated model needs one; no proxy discovery in a window that uses no network | App, library | Each has a test or a network log that fails before and passes after |
| 3 | **Approve each outward action specifically:** no *for this conversation* on one, the question shows what leaves and where, a link shows its address before it opens, and redirects and `http` are decided | `Mila::Agents`, app | Tests on the gate and the page; checked in the window |
| 4 | **Test the local promise:** no approval, no accepted connection, through a scripted transport | `Mila::Agents`, tests | A conformance test that fails with the check removed |
| 5 | **Protect what is at rest:** the search key in the credential store, then a design for encrypting conversations | App, spec | Key moved on all three platforms; design agreed |
| 6 | **Discuss and specify trust:** section 8's table, invariants and `../ToolCalling.md` open decision 2 | Spec | Todd's decisions recorded |
| 7 | **Labels on what enters a conversation,** starting with tool results, kept through storage and compaction | `Mila::Agents` | A label survives each path, by test |
| 8 | **Carry source records in the project store** when `../Apps/AssistantInterface.md` phase 4 builds it | App | The phase 4 schema holds them |
| 9 | **Read-only "find it" over past chats and a chosen folder:** "find the invoice from six months ago" | Spec, library, app | Measured: found, cited correctly, and honest when not found |
| 10 | **Saved sessions and several streams,** so work can continue beside the person | Library | Their own specs' gates |
| 11 | **Calendar read-only, then mail read-only** | App | Section 8's rules hold, by test, against hostile content |
| 12 | **Propose, and last Act,** per source | App | Each source's own approval and undo |

**How the Assistant's phases follow this order** (*proposed*, 2026-11-18). `../Apps/AssistantInterface.md` section 12
places its phases against these steps: its T1 is steps 2 to 4, T2 is steps 5 and 7, phase 4 is step 8, phase 5 is step 9
and phase 11 is step 10. Its sections 5.6 and 5.7 and `../Apps/AssistantOnboarding.md` 4.2 detail steps 2, 3, 6 and 7, and
Todd took their recommendations on 2026-11-18 ("Proceed with your recommendations").

## 12. How this governs the work

**Decided (Todd, 2026-11-18): everything we do from now on builds towards this.**

*Proposed as what that means in practice:*

- **A piece of work names what it advances:** a part of this vision, or the foundation it depends on. Library work
  counts fully. Family parity, deployment, quantization, backends and kernels are what make a capable local model
  possible on a person's own machine.
- **A change must not move away from the vision silently.** One that conflicts with it stops and is put to Todd, under
  `CLAUDE.md` *Stop and fix before working around*.
- **New designs say how they serve it,** and a spec that disagrees with this one is corrected or this one is.
- **Vocabulary follows it:** Mila Assistant is the product, not "the showcase".
- **Decisions move here.** When an open or proposed item is settled, its standing changes here in the same change, and
  the design that owns the area carries it (`../README.md` rule 1).

**Releases.** The scope and date of v0.21.0 are unchanged by this document. Whether any of section 11 enters it is
Todd's. Claude's recommendation is that v0.21.0 takes the small corrections (steps 1 and 2 at most) and no new theme.

## 13. Risks

- **One developer, two equally important halves.** `../Direction.md` section 7 already warns that the top of the stack
  draws the attention. Section 2's rule 2 is the guard.
- **The scope is vast.** Six areas of a person's life could absorb any amount of work. The per-source ladder and small
  steps are the bound.
- **The audience widens.** Ordinary people are not people who fight a toolchain or own a 16 GB card by choice. Hardware
  requirements, installation and the end-user prose rules matter more.
- **Safety may cost usefulness,** by an amount not yet known on local models.
- **A wrong claim is worse here.** A product whose identity is trust is damaged most by a promise it does not keep, so
  public language stays behind what tests show (`../Direction.md` section 2's truthful-today rule).
- **Model reliability.** A local model's errors at depth are measured facts (`../ContextProfile.md`), and they constrain
  what can be left unsupervised.

## 14. Open decisions

1. **The final form of the local promise.** Section 3 is Todd's current thinking.
2. **Trust:** how Mila treats downloaded documents and mail from known people (section 8). The person's own files start
   as content that may not direct Mila (`../Apps/AssistantInterface.md` 5.6, question E, decided 2026-11-18).
3. **Standing access:** how much may Mila read without being asked each time, beyond a folder the person named? A named
   folder starts read-only and is listed where it can be removed (`../Apps/AssistantInterface.md` 4.12, question G,
   decided 2026-11-18).
4. **The boundary.** *Decided (Todd, 2026-11-18):* labels and enforcement go in `Mila::Agents`, the store and connectors
   in the app (`../Apps/AssistantInterface.md` question F). `../Direction.md` 6.4's line on retrieval stores stands, and
   this is added beside it.
5. **How Mila files and organizes on its own.** *Decided for projects (Todd, 2026-11-18):*
   `../Apps/AssistantInterface.md` questions A and H.
6. **The words:** is *Your Workspace* too small a name for a person's digital world, and how is the product described
   publicly? ChatGPT's "personal intelligence environment" and "an intelligence layer between a person and their digital
   life" are candidates only.
7. **The audience statement** in the public message, once something ships that makes it true.
8. **A frontier model** as an opt-in resource: whether, when, and how it is presented under the local promise.
9. **What enters v0.21.0,** beyond this. *Decided (Todd, 2026-11-18):* the link that shows its address before it opens,
   from step 3, enters it. The rest of section 11 is still Todd's to place.

## 15. Relationship to other specs

- `../Direction.md` -- the library's trait, layering and release arc. Its 1.3 now points here for the product.
- `../Apps/AssistantInterface.md`, `../Apps/AssistantOnboarding.md`, `../MilaDesktop.md` -- how today's Assistant works.
  Their designs are the first stage of this vision. The first two hold their unbuilt work to it (round 3 and round 4).
- `../ToolCalling.md` -- the gate and grant that section 8 builds on.
- `../Agents/AgentTasks.md`, `../Agents/Compaction.md` -- long work and the labels it must keep.
- `../ContainedPrograms.md` -- programs held to their grant.
- `../ContextProfile.md`, `../Deployment/ModelSelection.md` -- what a model can be relied on for.
- `../Notebooks/MilaVisionDiscussion.md` -- the full discussion and its evidence. It is cited for evidence and never as
  the authority.
