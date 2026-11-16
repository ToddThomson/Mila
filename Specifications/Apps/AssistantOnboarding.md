# Mila Assistant Onboarding

**Status:** design, round 3 built (2026-11-15). Round 1 set out the facts and twelve questions. Round 2 is a
one-shot build at `0.21.0-dev+142`, which Todd asked for to see how Claude would design and build it ("test out how
you would design and implement this in a one shot attempt", Mac first). Every answer in questions 1 to 12 is
Claude's provisional one, marked as such, until Todd settles it. Round 3 (section 4.1) makes the first run a setup
assistant, with a licence accepted before any model: Todd took every proposed answer to questions 13 to 20 "to
start", to be tried and changed where it needs to be (2026-11-15), and it is built at `0.21.0-dev+146`.

What a person meets from Mila Assistant's first launch to its first reply, and how the model Mila suggests for
their computer is chosen. The suggestion is the planner's answer, never the app's (Todd, 2026-11-13).

---

## 1. What this decides

- The first run: what the window shows when nothing is installed, in what order, and what the person does at
  each step until the home screen of `MilaDesktop.md` 3.4.
- The suggestion: which published model or models Mila offers for this computer, ranked how, and what in the
  library answers that.
- What the person can do instead of the suggestion, and what happens when there is no suggestion to make.

It does not decide the screens for a store that has models but none loaded; those are built (`MilaDesktop.md`
3.4, `+141`) and are the fallback this design replaces only on a first run.

## 2. Where it stood before round 2

**The window.** A fresh store has no model (CLAUDE.md). Since `+141` the window then showed *Install a model to
start*, the reason it asks, and *Choose a model to install*, which opened Settings at *Models* and fetched the
published list. That was the catch for an empty store, not onboarding: it named no model.

**The published list.** `publishedModels()` (`Apps/Desktop/Session/Src/Desktop.Models.ixx`) listed the
publisher's repositories and fetched each one's manifest, keeping only its size and its weight format. The manifest
carries more (`Lib/Src/Distribution/Packaging/ModelManifest.ixx`): the architecture and its version, the minimum
Mila version, the licence, the publisher's sampling, every file with its bytes and digest, and `ModelCapabilities` --
`instruct`, `thinking`, `tools`, `agentic` and the input modalities. `agentic` is Mila's own measured verdict,
written only for a model whose context profile was measured (`ModelCapabilities.ixx`, `ContextProfile.md`).

**The planner.** Each family's `planDeployment( path, request )` reads the weights header -- the model's
configuration rides in it, in `__metadata__["mila_config"]` (`Dnn/Serialization/WeightsReader.ixx`) -- then
constructs the graph on each allowed device, takes one reading of it and plans; nothing fitting is a refusal that
names why, not an error (`Deployment.md` 5 and 6). Pricing allocates nothing (`Deployment.md` 3.1). Every entry took
a package on disk: nothing planned a model before it was installed. Since `+138` the planner on a Mac reads the GPU
memory other applications hold (`Deployment.md` 12.10).

**The session.** An install into a session with nothing resident loads the model it installed
(`Desktop.SessionProcess.ixx`, `installModel`), so a first install already ends with a model loaded.

## 3. Rules it stands on

- **The suggestion comes from the planner** (Todd, 2026-11-13): it plans each candidate on this computer. A
  rule of the app's own -- a model's download against the machine's memory, a family's name -- is not one.
- **An app adds no capability to the library** (CLAUDE.md). Whatever the suggestion needs that the library
  does not have is library work.
- **Suitability is measured:** what a model can do is what its manifest declares, never inferred from its family
  or name (`ModelCapabilities.ixx`).
- **A model runs in the format it was trained for**, and the weight format is never the planner's choice
  (`Deployment.md` 12.1): the suggestion ranks packages as published.
- **End-user prose** (CLAUDE.md): every screen of the first run is read by someone who has never heard of Mila.

## 4. The first run, as built at +142

Section 4.1 replaces step 1 and *Skip* at `+146`: the welcome and the licence are panes of their own before steps 2
to 6, which are its model pane.

1. With nothing installed and nothing loading, the empty conversation is one screen that welcomes and asks: Mila's
   mark, *Welcome to Mila* and Mila's byline *Local LLMs on hardware you own -- at the metal* (Todd, 2026-11-14;
   the wording is being chosen across every surface). *Skip* in the window's corner on every step passes the
   welcome by for this run of the window: the window returns with *Install a model to start* and *Choose a model to
   install*, and the welcome shows again at the next launch while nothing is installed.
   It has the whole window (Todd, 2026-11-14): no sidebar, rail or header, as there is nothing yet to go back to,
   and no message box. They return as the installed model loads.
2. The screen asks the session for the published list as it first shows (`models.refresh`, online) and says
   *Finding the model that suits this computer* until the list comes back.
3. One line introduces it, *To start, install the model Mila suggests for this computer*, then the suggestion in a
   card: its name for a person to read, the manifest's `display_name` (*Gemma 4 12B*), or its name when the
   manifest declares none (Todd, 2026-11-14), what it can do (*Works through tasks with
   tools* only when it is agentic here, section 5; else *Answers questions and holds a conversation*; then whether
   it thinks and what it reads besides text), *Runs on this computer with room for long tasks* when it is agentic
   here, and its licence by name, then one *Install*, centred, as the card's last line (Todd, 2026-11-14). The
   download's size and the conversation length are Settings' to show, not the first run's. *Choose another model*
   sits below the card, on its left edge, and goes once *Install* is pressed.
4. The install's progress replaces the button. The session loads the model when the install ends; the lines the
   install and the load say are not added to the empty conversation, so the home screen follows at once.
5. *Choose another model* opens Settings at *Models*, where each published model carries what the planner said:
   *Suggested for this computer*, *Runs on this computer*, or why it does not.
6. When the publisher cannot be reached: the reason, and *Try again*. When it answers and nothing runs here:
   *None of the models published for Mila runs on this computer*, why for the first model the planner ranked, and
   *See every model*.

The page is `Apps/Desktop/Ui/views/transcript.js` (`renderFirstRun`), its state `state.js` (`firstRunStep`,
`suggestedModel`), its words `format.js` (`modelAbilities`, `suggestionFacts`). The mock plays it as scenario 8 and,
with the publisher unreachable, 12.

## 4.1 The first run as a setup assistant: welcome, licence, model (round 3, built at +146)

Todd, 2026-11-15: Mila Assistant does not reach a person without a licence they have accepted (as is, no
liability), and the first run follows Apple's setup assistant: welcome, licence, model.

**What Apple's setup assistant guidelines say.** Today's HIG has no page for it. The rules are in the Mac's
*Aqua Human Interface Guidelines* (June 2002, chapter 14, *Setup Assistants*, pages 241-243; the 2004 edition drops
the section and keeps only the glossary entry). Read from the archived PDF, 2026-11-15:

- A setup assistant guides people through a product's setup, opens on its own when appropriate (the first time the
  application opens, for example), and is ideally used once.
- **Go Back** and **Continue** move between panes.
- The first pane is titled *Introduction* and explains the purpose of the panes that follow.
- The last pane is titled *Conclusion*: it says what changed and how to change it later, with a default **Done**
  and a dimmed **Go Back**.
- In most cases, one question per pane.
- Feedback where it helps; a progress bar may sit to the left of **Go Back**.
- While it is open, only the application menu (About, Quit) and the Edit menu; no Help menu or help button, as the
  assistant is the help. The title bar's close and zoom buttons are dimmed and minimize is available.
- It does not fill the screen: the rest of the system stays reachable.

The same guidelines name accepting a licence agreement as a modal step a single-window utility asks for (chapter
on sheets). Today's HIG *Onboarding* page (read 2026-11-15) adds two things that still bear on this: agreements are
the App Store's to show, else included in a balanced way that does not disrupt; and setup waits for defaults unless
the app cannot work without it. Mila Assistant is not in the App Store, and cannot work without either step.

**The panes.** The window keeps its size; the panes fill it, as section 4 step 1 does. Each pane ends in **Go Back**
and its forward button at the bottom right, the forward button the default, and the progress of what the pane is
doing on the left beside them, as the guidelines lay them out. There is no *Skip*: a setup assistant has none, and
two of the three steps cannot be passed by. Within the window the panes are the one way forward: no message box,
and the new-conversation and Settings shortcuts do nothing.

1. **Introduction: Welcome.** The mark, *Welcome to Mila*, the byline, and what follows: *Two steps to start:
   accept Mila's licence, then install the model that suits this computer*; with models already installed, *To
   start, accept Mila's licence*; with the licence accepted at an earlier launch, *To start, install the model that
   suits this computer*. **Continue**; **Go Back** dimmed. Nothing is asked of the network yet.
2. **Licence.** Titled *Licence*, then one line: *Mila is yours to use under this licence. Read it to the end to
   accept it.* The text in a box that scrolls, and keeps the place read to. **Accept** in **Continue**'s place,
   enabled once the end of the text has been in view, which a text that fits without scrolling meets at once; a
   licence accepted earlier on this run shows **Continue**. **Go Back** returns to the welcome, and **Quit** declines
   (question 13). A refused acceptance is said on this pane. A package whose licence cannot be read says so here, with
   **Go Back** and **Quit** and no way past.
3. **Model.** Titled *A model for this computer*, then section 4's steps 2 to 6 as built, the publisher asked when
   this pane first shows. **Install** is the pane's forward button, enabled once a model is suggested; the install's
   progress, then the load's, sits on the left while they run, and **Go Back** and **Install** are dimmed until they
   end. *Choose another model* stays below the card. **Go Back** returns to the licence when it was read on this run,
   else to the welcome. A model installed and then refused at its load leaves the setup for *Choose a model to start*,
   which says why.
4. **Conclusion** (question 18, not one of Todd's three). *Mila is ready*, *<the model's display name> is installed
   and loaded*, and *To change the model, open Settings, then Models*. **Done**, default; **Go Back** dimmed.
   **Done** opens the home screen. It follows an install begun on the model pane, Settings' included.

With models installed and the licence not yet accepted, the welcome and the licence show, and once it is accepted the
session opens the remembered model: the window returns at once.

While the panes show, a packaged Mac's menu bar is the application menu (About, Quit) and Edit only (question 19).
The page tells the window (`setup.showing`), which builds the menu.

**What is accepted, and who keeps it.** The session decides, not the page:

- The licence text ships in the package beside the session, the same `License.md` the repository carries
  (`Desktop.MilaLicense`; the build copies it, the `mila-session` install component carries it). A package
  without it, or with an empty one, starts and says the licence is missing; nothing can be accepted.
- The session records an acceptance in its own settings, `settings.json`'s `license`: the text's SHA-256 digest and
  the time, in UTC. A package whose licence text has a different digest asks again at the next launch, and an
  acceptance the file holds in a form that cannot be read is no acceptance.
- Until the licence is accepted the session refuses `model.install`, `model.load` and a turn, and does not open the
  remembered model, so a front end other than this page meets the same refusal. The page learns where the person
  stands from the session's `license` event, the reply to `ready`, and accepts by the digest of the text it showed
  (`license.accept`), so an acceptance is always of the text that was read (`MilaDesktop.md` 4).

**Two licences.** Mila's own, accepted once, here. Each model's (Gemma's terms, Llama's community licence) at its
install, question 16.

**The installers.** Checked in the electron-builder the app uses (26.15.3, `app-builder-lib/scheme.json` and its NSIS
templates): `dmg.license` shows a licence agreement when the DMG is mounted (`.rtf`, `.txt`, `.html`), and
`nsis.license` adds a licence page, which the templates insert in the one-click installer as well
(`oneClick.nsh`). Linux packages as a folder (`dir`), with no installer to carry one. The proposal is the in-app step
as the one acceptance: the same on all three systems and recorded. The installers' licence pages are question 15.

**Not Claude's to say:** whether a click-through adds legal force, and the licence's wording. Those are counsel's.

The page's pieces: `setupPane` in `state.js` says which pane shows, from the session's `license` and the page's own
`setup.continue`, `setup.back` and `setup.done`; `renderSetup` in `views/transcript.js` draws it; `welcome.skip` is
gone. The mock carries the licence as the session does: scenarios 8 and 12 begin with it not accepted, and 13 has
models installed and the licence not accepted.

## 5. The suggestion, as built at +142

**Shape A, fetch what planning reads.** Of round 1's three shapes, A is built: it plans the very bytes the install
will download, and needs no manifest change. B would copy into the manifest what the weights already carry; C plans
on the publisher's machine, which section 3's first rule rules out.

The library pieces, all in `Lib/Src`:

- **A header alone is a weights file the planner reads.** `WeightsReader` opens a safetensors file that ends
  exactly where its header does: everything the header declares reads as for the whole file, no tensor's bytes do,
  and `readTensorBlob` and `streamTensorBlobs` refuse it (`holdsHeaderOnly`). A file that ends at its header is
  never a whole weights file, so the form cannot be mistaken for one. The families' planners are unchanged: a
  plan reads only the header.
- **A part of a file.** `HttpRequest::last_byte` sends `Range: bytes=first-last`; a 200 is `RangeIgnored`, and a
  server that ignores the range is stopped at the first byte past the part rather than buffered.
  `IModelHub::fetchFilePart` reads one, and `HuggingFaceHub` implements it.
- **`Distribution::PublishedModel`**: one repository and its manifest, `list( hub, owner )` (what
  `Desktop.Models.ixx` did by hand), `downloadBytes()`, and `weightsHeader( hub, store )` -- two small reads, the
  length and the header, written to `<store>/headers/sha256-<weights digest>.header` and read from there after.
  A republished model's digest changes, so it is read again.
- **`ModelHandle::check( PublishedModel )` and `ModelHandle::planFor( PublishedModel, request, hub, store )`**:
  what the installed-model forms answer, from the manifest and the header, with the device type chosen as `open`
  chooses it. `OpenRefusal::Reason::ManifestUnread` is new.
- **`Deployment::planPublishedModels( hub, owner, request, store )`** plans every published model and ranks them
  (`PublishedModelFit`): those that fit and take turns first, agentic here before the rest, the larger download
  first within each; then those that fit and do not take turns; then the rest by name. *Agentic here* is the
  manifest's verdict and a best plan at least `kAgenticContextLength` (65536) long, the depth the verdict is
  measured at (`ContextProfile.md`). The suggestion is the first that fits and takes turns.

The request is the one a load sends (`requestFor`: the session's devices, the context automatic unless the person
fixed one), so the plan is the one the load after the install will meet on a device in the same state.

## 6. Questions, with round 2's provisional answers

Each answer below is Claude's, built at `+142` so it can be looked at; none is settled until Todd says so.

1. *Where planning before install lives, and in which shape.* Shape A, in the library: section 5.
2. *What fits means.* The planner's own answer for the load's request, against the memory free now, or the whole
   card when a model is resident. An agentic model counts as agentic here only at 64K or more.
3. *How the models that fit are ranked.* Agentic here first, then the larger download. **The larger download is
   the weakest answer in the build:** nothing a manifest carries says which of two fitting models answers better,
   and size stands in for that unmeasured. The replacement is measured quality read by the planner (Todd,
   2026-11-13: the cards' measurements are to be the planner's too), designed in `../Deployment/ModelSelection.md`;
   the suggestion becomes the first model ranked there for agentic work.
4. *One suggestion or a ranked few.* One, with *Choose another model*.
5. *When Mila suggests.* On the first run, and as a mark in Settings' *Models* whenever the published list is shown.
   No alert when a model is published.
6. *Whether the first run loads after the install.* It loads, as the session already did.
7. *Two GPUs.* The planner plans every allowed device and its best plan is the answer; the app chooses nothing.
   Not run on a two-GPU machine.
8. *The download.* Its size before it starts. Free disk space is not checked first.
9. *Per activity or overall.* One overall.
10. *Before the suggestion.* *Welcome to Mila* and one line. Round 3 replaces it: a welcome, then the licence
    (section 4.1).
11. *None fits, or no publisher.* Section 4, step 6.
12. *The licence.* Shown by name. No agreement wording: what installing means for the licence is Todd's. Round 3
    takes Mila's own licence to section 4.1; the model's is question 16.

**Round 3's questions (section 4.1).** Todd took every answer below to start, to be tried and changed where it needs
to be (2026-11-15):

13. *Declining Mila's licence.* Quit, from the menu or a **Quit** beside **Go Back** on the licence pane; nothing is
    recorded, and the next launch opens the assistant again at the welcome.
14. *A step indicator.* None: the guidelines name none, and each pane says what it asks.
15. *The installers' licence pages.* Not added: the in-app step is the one acceptance, so a person is not asked
    twice on the Mac and Windows and not at all on Linux.
16. *A model's licence at its install.* Its name on the card becomes a link that opens the text. Whether *Install*
    says the person accepts it, and in which words, is counsel's and Todd's. **Not built at +146:** `models.list`
    carries the licence's name only, not where its text is, so the link needs the published model's licence
    address in the list first.
17. *Which text.* Mila's MIT licence as it stands, or one written for the app's end users. Counsel's.
18. *A Conclusion pane.* The guidelines end on one; Todd's three steps do not. Proposed: add it, since today the home
    screen follows the load with nothing said about where the model is changed.
19. *The 2002 window rules.* The reduced menu bar: yes, on a packaged Mac. Windows and Linux packages have no menu
    bar at all (`Window/src/main.js`), so there is nothing to reduce. The dimmed close and zoom buttons: no, since a
    person who closes the window has declined, which quitting already covers, and Windows has no dimmed close.
20. *The model pane when the publisher cannot be reached, or nothing runs here.* Section 4 step 6's screens, with
    **Go Back** to the licence; nothing past the model pane opens until a model is installed, and the window with
    *Install a model to start* is no longer reached from the first run.

**Open, found in the build:**

- **Planning with a model resident** uses the request's whole-card form (`withWholeDeviceMemory`, `Deployment.md`
  12.5), as a listing does, so the resident model's memory does not count against the others. A shortfall then
  still quotes the free memory in its sentence; the wording for the whole-card case is not written.
- **The time to plan every published model is not measured.** Each costs two small reads of its weights the first
  time and a graph construction on the device every time.
- **Disk space** before an install (question 8).
- **Two formats of one model share a display name** (*Gemma 4 12B* in Q4_0 and in FP4). The first run shows one
  suggestion, so it is untouched; Settings' *Models* is where they meet: one row per model with Mila choosing the
  format, or one row per package with the format in plain words. Todd's call, the next round after the first run.
  The first-run card, the header's model menu, the choose-a-model rows and the session's *Installing*, *Loading*
  and *Installed* lines use the display name; Settings' *Models* still shows the name.

## 7. Found while scoping, fixed at +142

- An install's and a load's own lines (*Installed ...*, *License ...*, *Model loaded.*, *Context ...*) filled the
  empty conversation after a first install and hid the home screen. They now carry `during` (`"install"` or
  `"load"`), and the page does not add them to a conversation that has not begun.
- A failed fetch of the published list was an `error` with no `during`, so it landed in an empty conversation. It
  is now `during: "models"`, said on the screen that asked for the list, and asking again clears it.

## 8. Not in this design

- Choosing a model per conversation or per activity after the first run (`MilaDesktop.md` 3.2's library query,
  a proposal for `Mila::Agents`).
- The screens for a store with models and none loaded (`MilaDesktop.md` 3.4).
- The renaming of the app and its folder to Mila Assistant.
