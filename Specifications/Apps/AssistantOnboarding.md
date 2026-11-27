# Mila Assistant Onboarding

**Status:** design, round 3 built (2026-11-15). Round 1 set out the facts and twelve questions. Round 2 is a
one-shot build at `0.21.0-dev+142`, which Todd asked for to see how Claude would design and build it ("test out how
you would design and implement this in a one shot attempt", Mac first). Every answer in questions 1 to 12 is
Claude's provisional one, marked as such, until Todd settles it. Round 3 (section 4.1) makes the first run a setup
assistant, with a licence accepted before any model: Todd took every proposed answer to questions 13 to 20 "to
start", to be tried and changed where it needs to be (2026-11-15), and it is built at `0.21.0-dev+146`. Round 4 (section
4.2, 2026-11-18, taken by Todd the same day): Mila contacts the model hub only when the person asks, to keep the local
promise of `../Direction/MilaVision.md` section 3. Round 5 (2026-11-22, Todd, from the first run he made himself):
the panes become a sheet of one size in an empty window, the workspace starting only when the setup ends; the welcome
loses its line about what follows, and the first and last panes lose their dimmed **Go Back**; the panes are titled by
name and say *License*; an accepted License is done, with no way back to it; and *Install Model* asks the model hub
as it shows, reversing round 4 for the setup alone (sections 4.1 and 4.2). Round 6 (2026-11-23, Todd, installing Qwen
3.8 27B: "There is no size, etc. The Model card should show the download and progress. Can the download be
stopped?", then "sizes only, cancel button"): the card says its download's size, the download's bar and how much has
come sit on the card with **Cancel**, and a cancelled install continues from where it stopped; then, after an install
refused by a lock a closed session left ("this failed. And then it is poor UX"), closing Mila stops an install and
the store's lock ends with its process (section 4, steps 3 and 4). Round 7 (2026-11-23, Todd, the same walk): no
*Starting the download*, "the progress bar is the needed UI", so the bar shows from the press and moves while the
store checks what an earlier download kept; **Cancel** deletes what arrived, as the person asked to cancel, while
closing Mila keeps it; *Choose another model* goes and **Skip** comes, "if a model is not installed then the user will
need to select the avail list through normal channels"; and an update of Mila is to have a *What's New* pane
(section 4.3, not built). Round 8 (2026-11-24, Todd, decided in part, not built): an onboarding model for every
backend, its download begun at the first launch, Mila's terms accepted before the first chat, and the interview as the
workspace's first chat (section 4.4). Where it is decided it supersedes rounds 3, 5 and 7's model pane. Round 10
(2026-11-25, Todd, walking the interview): its two goals, and that it ends in the person's decision to stay on the
onboarding model or move to a regular one (section 4.4). Round 13 (2026-11-26, Todd, after walking round 12): one
download, of the model that suits this computer, and the interview runs on it; the onboarding model and the second
download go (section 4.4).

What a person meets from Mila Assistant's first launch to its first reply, and how the model Mila suggests for
their computer is chosen. The suggestion is the planner's answer, never the app's (Todd, 2026-11-13).

---

## 1. What this decides

- The first run: what the window shows when nothing is installed, in what order, and what the person does at
  each step until the workspace overview of `AssistantInterface.md` 4.3.
- The suggestion: which published model or models Mila offers for this computer, ranked how, and what in the
  library answers that.
- What the person can do instead of the suggestion, and what happens when there is no suggestion to make.

It does not decide the screens for a store that has models but none loaded; those are built (`MilaAssistant.md`
3.4, `+141`) and are the fallback this design replaces only on a first run.

## 2. Where it stood before round 2

**The window.** A fresh store has no model (CLAUDE.md). Since `+141` the window then showed *Install a model to
start*, the reason it asks, and *Choose a model to install*, which opened Settings at *Models* and fetched the
published list. That was the catch for an empty store, not onboarding: it named no model.

**The published list.** `publishedModels()` (`Apps/Assistant/Session/Src/Assistant.Models.ixx`) listed `mila-llm`'s
repositories on the model hub and fetched each one's manifest, keeping only its size and its weight format. The manifest
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
(`Assistant.SessionProcess.ixx`, `installModel`), so a first install already ends with a model loaded.

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
- **Nothing is contacted until the person asks** (*proposed*, section 4.2; the vision's step 2). First-run screens claim
  nothing about privacy that tests do not yet show (the vision's section 13).

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
   here, *Download: <size>* from the published list's `size_bytes` (round 6), and its licence by name, then one
   *Install*, centred, as the card's last line (Todd, 2026-11-14). The conversation length is Settings' to show, not
   the first run's. Round 7 took *Choose another model* from below the card (step 5).
4. While the download runs, the card shows its bar, how much has come (*3.1 GB of 15.2 GB*) and **Cancel** (round 6;
   sizes only, no time left, Todd, 2026-11-23). The bar is the whole model, every file the manifest lists, not each
   file in turn. It shows from the press of **Install**, moving with no amount until the session reports one (round
   7, replacing *Starting the download*). When an earlier download of the model was interrupted, the store first reads
   back the bytes it kept, to continue their checksum, and reports them as it goes, so the bar runs quickly up to what
   was kept and goes on from there (`ModelDistribution.md`, *Resume*).
   **Cancel** stops the download and deletes what arrived (round 7, Todd: "The user has said to cancel the
   download"): the model's partial files and the files of it that finished, except any an installed model also uses.
   The card returns with **Install**, and nothing more is said. Closing Mila stops the download and **keeps** what
   arrived, as does a connection that fails, and installing it again continues from there: the window ends the
   session's input and waits up to 5 seconds for it to end (Todd, 2026-11-23, after a session ended mid-download left
   the store's lock behind and the next install was refused with the library's words; the lock is now the operating
   system's, `ModelDistribution.md`). There is no **Pause**. Another Mila program installing the same model is said as
   *Another Mila program is installing <model>. Install it again once it has finished.* A file already in the store
   downloads nothing, so its bytes are never counted and the bar ends short of full; the install ending closes it.
   Settings' *Models* shows the same on the model's row. The session loads the model when the install ends; the
   lines the install and the load say are not added to the empty conversation, so the overview follows at once.
5. **Skip** (round 7) ends the setup with no model. The workspace then shows *Install a model to start* and *Choose a
   model to install*, which opens Settings at *Models*, where each published model carries what the planner said:
   *Suggested for this computer*, *Runs on this computer*, or why it does not. At the next launch with nothing
   installed the setup shows *Install Model* again.
6. When the model hub cannot be reached: the reason, and *Try again*. When it answers and nothing runs here:
   *None of the models published for Mila runs on this computer*, and why for the first model the planner ranked.
   **Skip** is the way on from either.

The page is `Apps/Assistant/Ui/views/transcript.js` (`renderFirstRun`), its state `state.js` (`firstRunStep`,
`suggestedModel`), its words `format.js` (`modelAbilities`, `suggestionFacts`). The mock plays it as scenario 8 and,
with the model hub unreachable, 12.

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

**The panes.** The panes are a sheet over the window (Todd, 2026-11-22: "a layer ( sliding dialog/window pane -
always the same size ) on top of the main window"). The sheet is one size for every pane, 620 by 500, and slides
down from the top of the window as it opens. The window behind it is an empty shell: no sidebar, header or pane. The
workspace starts only when the setup ends (Todd, 2026-11-22: "The onboarding runs over an empty shell to start. Once
onboarding has completed the main UI is then started"). Until the session has said whether the setup is due -- where
the licence stands and, with it accepted, whether a model is installed -- the window stays empty, so the workspace is
never drawn only to be covered; a session that stopped is the exception, said in the workspace. The window's minimum
size, 760 by 600, holds the sheet with its margins. Inside the sheet, the licence's box and the
model pane scroll. Each pane ends in its forward button at the bottom right, the default and holding the keyboard
when the pane shows, with **Go Back** before it wherever there is a pane to go back to, and the progress of what the
pane is doing on the left. The first and last panes have no **Go Back**: a button that can never be used is not shown
(Todd, 2026-11-22, where the guidelines dim it). The welcome and the License have no *Skip*: a setup assistant has
none, and those two steps cannot be passed by. *Install Model* has one since round 7. The panes are the one way forward: the new-conversation, Search and Settings shortcuts do
nothing. The page draws the sheet, so it is the same on Windows, macOS and Linux.

Each pane is titled with a name, not a sentence (Todd, 2026-11-22): *Welcome*, *License*, *Install Model*, *Ready*.
What a person sees says *License*, the spelling Todd chose that day; the code's identifiers and these notes keep theirs.

1. **Welcome.** The mark, *Welcome* and the byline. **Continue**. No line about what follows: *To start, accept
   Mila's licence* told the person to do what they could not do on this pane (Todd, 2026-11-22), and **Continue**
   leads to it. Nothing is asked of the network yet.
2. **License.** One line: *Mila is yours to use under this license. Read it to the end to accept it.* The text in a
   box that scrolls, and keeps the place read to. **Accept** in **Continue**'s place, enabled once the end of the text
   has been in view, which a text that fits without scrolling meets at once. There is no **Go Back** (Todd,
   2026-11-25: "Remove the Go Back"), since the welcome before it holds nothing to change, and no **Quit** (Todd, the
   same day: "The T&C 'Quit' seems abrupt. Can the main screen X be sufficient here?"): declining is closing the
   window, which quits Mila on Windows, macOS and Linux alike and records nothing. Round 3's **Quit** (question 13)
   is reversed; so is it on *Getting Mila Ready* and on the full pane. A refused acceptance is said on this pane. A package whose licence cannot be read
   says so here, with **Go Back** and **Quit** and no way past.
   **Accepted is done** (Todd, 2026-11-22): the setup moves past the License and nothing leads back to it or to the
   welcome. A setup that finds it accepted on an earlier run, with nothing installed, begins at *Install Model*. The
   text stays readable from About.
3. **Install Model.** Section 4's steps 2 to 6 as built, the model hub asked as the pane first shows (section 4.2),
   with *Finding the model that suits this computer* while it answers. **Install** is the forward button, enabled
   once a model is suggested, with **Skip** before it (round 7, Todd, 2026-11-23, reversing "no Skip for now" of
   2026-11-22), both dimmed while a download or a load runs; the download's progress and **Cancel** are on the card
   (section 4, step 4), and the load after it sits on the left while it runs. No **Go Back**, since the License
   before it is done. A model installed and then refused at its load leaves the setup for *Choose a model to start*,
   which says why.
4. **Ready** (question 18, not one of Todd's three). *<the model's display name> is installed and loaded*, and *To
   change the model, open Settings, then Models*. **Done**, default, and no **Go Back**. **Done** opens the overview.
   It follows an install begun on the model pane, Settings' included.

With models installed and the licence not yet accepted, the welcome and the licence show, and once it is accepted the
session opens the remembered model: the window returns at once.

While the panes show, a packaged Mac's menu bar is the application menu (About, Quit) and Edit only (question 19).
The page tells the window (`setup.showing`), which builds the menu; the empty window before the setup counts as it.
A run from the source folders keeps Electron's menu on Windows and Linux for its shortcuts -- developer tools, reload,
zoom -- with its bar hidden, so the window looks as a released Mila's does: over the panes since 2026-11-22, and
everywhere since 2026-11-25 (Todd: "hide it in the development run too"). Alt shows the bar, except while the panes
show.

**What is accepted, and who keeps it.** The session decides, not the page:

- The licence text ships in the package beside the session, the same `License.md` the repository carries
  (`Assistant.MilaLicense`; the build copies it, the `mila-session` install component carries it). A package
  without it, or with an empty one, starts and says the licence is missing; nothing can be accepted.
- The session records an acceptance in its own settings, `settings.json`'s `license`: the text's SHA-256 digest and
  the time, in UTC. A package whose licence text has a different digest asks again at the next launch, and an
  acceptance the file holds in a form that cannot be read is no acceptance.
- Until the licence is accepted the session refuses `model.install`, `model.load` and a turn, and does not open the
  remembered model, so a front end other than this page meets the same refusal. The page learns where the person
  stands from the session's `license` event, the reply to `ready`, and accepts by the digest of the text it showed
  (`license.accept`), so an acceptance is always of the text that was read (`MilaAssistant.md` 4).

**Two licences.** Mila's own, accepted once, here. Each model's (Gemma's terms, Llama's community licence) at its
install, question 16.

**The installers.** Checked in the electron-builder the app uses (26.15.3, `app-builder-lib/scheme.json` and its NSIS
templates): `dmg.license` shows a licence agreement when the DMG is mounted (`.rtf`, `.txt`, `.html`), and
`nsis.license` adds a licence page, which the templates insert in the one-click installer as well
(`oneClick.nsh`). Linux packages as a folder (`dir`), with no installer to carry one. The proposal is the in-app step
as the one acceptance: the same on all three systems and recorded. The installers' licence pages are question 15.

**Not Claude's to say:** whether a click-through adds legal force, and the licence's wording. Those are counsel's.

The page's pieces: `setupPane` in `state.js` says which pane shows, from the session's `license` and the page's own
`setup.continue`, `setup.back` and `setup.done`; `isStarting` says the window is still empty; `views/setup.js` draws
the sheet; `welcome.skip` is gone. The mock carries the licence as the session does: scenarios 8 and 12 begin with it not accepted, and 13 has
models installed and the licence not accepted.

## 4.2 Contacting the model hub only when asked (round 4)

**Round 5 takes the setup's pane out of this** (Todd, 2026-11-22: "We do not need to tell the user what we are about
to do - we need to just do it", then "just remove that step"). *Install Model* asks the model hub as it first shows,
with no explanation and no **Look for models**, once on a run; **Try again** asks again when it could not reach it.
What the rest of this section holds still holds for the setup's request: no token goes with it, and the planning stays
on this computer. Settings' *Models* is unchanged and still waits for **Look for models**. Below is round 4 as
written, for Settings and as the record of what the setup did from `+155` to round 5.

*Until `AssistantInterface.md` phase T1* the model pane asked the model hub as it first showed, and the way to Settings'
*Models* did with no list. That sent this computer's address to HuggingFace unasked, with a HuggingFace token if one was
found on the computer, tying the request to the person's account (`../Notebooks/MilaVisionDiscussion.md` 8, G4).

**The model pane opens without contacting anyone** (round 4; the setup's no longer does). It says what looking
involves, in words for someone new to all of this (a draft, Todd's to change):

> To suggest a model, Mila asks Hugging Face, where its models are published, what is available. Hugging Face sees this
> computer's internet address and which files Mila reads. Nothing you have written is sent, and Mila works out here what
> fits this computer.

**Look for models** is the pane's forward button. Pressed, section 4's steps 2 to 6 follow, and **Install** takes its
place once a model is suggested. **Go Back** is as section 4.1 has it.

- **Settings' *Models*** shows what is installed without contacting anyone. *Available to install* holds **Look for
  models** until it is pressed in this run of the app. The list is fetched again only by pressing.
- **Pressing *Install*, with the model named,** is the approval for its download.
- **No token goes with a listing or a download** (`makeAnonymousModelHub`). A token found on the computer is to be used
  only for a model that needs one, after the person is told so and agrees. **That step is not built:** until it is, a
  gated model cannot be installed from Mila Assistant.
- **The planning stays on this computer.** Every published model's header is read, whichever would fit, so the requests do
  not reveal what this computer has.

*Checked by:* a page test that no `models.refresh` is sent before the press, on Settings (and, until round 5, the setup's pane); a
scripted hub showing no `Authorization` header on a listing or a public download; and the window's network log on Windows
and macOS showing no connection before the press. On Linux the session's tests run in WSL and the window is unverified.
`mila-store` and Chat are commands a person runs to do this very thing, and are not changed here.

## 4.3 An update: What's New (decided, not built)

Todd, 2026-11-23: "although it does not apply right now the onboarding for an update should have a 'What's New'
pane". Nothing has been updated yet, since Mila Assistant has not shipped. Once it has, the first launch of a newer
version shows a *What's New* pane in the same sheet, before the workspace starts.

Proposed, Claude's, for Todd: **Continue** as its one button, and a licence whose text changed asked for again on the
pane after it, as section 4.1 has it. Open, to settle when it is built: which versions show it (a minor, a patch); where its words come from (the GitHub
Release body is a developer's summary of the minor, and the pane is read by an end user); and where the last version
seen is kept, which the session's settings would hold beside the licence's acceptance.

**Round 9 (Todd, 2026-11-25), decided, not built: one path for a new install and an update.** "The onboarding process
is launched for new installs and updates. The onboarding always follows a set path: Welcome, license (the user
chooses to accept or quit), steps to get mila ready, steps to describe capabilities or new capabilities." On it:

- **The licence is asked for only when acceptance is required**: on a new install, and on an update whose text
  changed (section 4.1's digest). Its step stays in the path and is passed over otherwise.
- **The capabilities are said for this computer**, from what is measured of the installed model here (agentic here,
  or the conversation alone), so a smaller machine's onboarding describes the assistant it gets ("I hadn't thought
  about a filter based on the compute env. Nice"). On an update, the step says what is new.
- **The download is asked for, on the *Getting Mila Ready* step after the licence** (Todd, the same day, "Go with
  (b)"), answering his second thoughts about downloading without the person's consent; accepting the terms is not
  taken as consent to it ("a user does not associate the acceptance of T&C as also consent to allow the download").
  The step says, in one plain line, how much space Mila needs on this computer, with no model named, and its button
  starts the download; declining is Quit. This revises round 8's download at launch.
- **The size is exactly what will arrive**: the model hub is listed and the onboarding model picked for this computer
  (`onboardingModel`, built in round 8) before the question, a few small reads sent with no account. Nothing is
  downloaded until the person agrees; section 4.2's promise is said that way.
- **The capabilities step shows while the download runs, and the interview follows on a new install only** (Claude's,
  taken by Todd the same day: "Yes to both, then build round 9").

**Built, Claude, 2026-11-25, on Windows.** The session lists the model hub and picks at launch and downloads only on
`onboarding.download`, which is refused before the terms; the agreement is kept in the settings (`download_agreed`), so
a download stopped by quitting goes on at the next launch. The interview waits for `onboarding.panes_done` as well as
the terms and the model. `last_version` is the version whose onboarding the person went through; on a newer one,
`Prompts/whats-new.txt`, when the release ships one, is `session.state.whats_new`, and a version with nothing new is
counted as gone through at launch. The page's path is Welcome, Terms and Conditions when acceptance is required,
Getting Mila Ready when the download is not yet agreed (the space Mila needs, no model named, **Download** and
**Quit**), What Mila Can Do (from the onboarding model's measured capabilities here) or What's New on an update, and
the full Getting Mila Ready pane only when the person reaches the end first; a ring at the sheet's corner shows the
download and the start on every pane while they run. Run once through `mila-session` with a script in the page's
place, the local model hub and a throwaway Mila folder, on Gemma 4 12B Q4_0: nothing downloaded before the agreement
at 4.4 s, the download ended at 27 s, the interview opened at 30 s after the panes were done, and `remember_name` and
`finish_setup` ran. The mock's scenario 21 walks the same panes. The full pane is titled *Almost Ready* (Todd, 2026-11-25), as
*Getting Mila Ready* is the step that asks for the download, and is a simple list with checks (Todd, the same day: "A
simple list with checks would do here"): what Mila needs downloaded, Mila started, the first chat
opened, each checked once done, the window's ring still showing the download. The Welcome begins with Mila's lockup,
the mark and the wordmark as one object as on the website, with *Welcome* beneath it as the pane's title (Todd, the
same day). Found in Todd's first walk and fixed:
the panes were drawn again for every progress message, which replaced their buttons between a press and its release,
so Continue did nothing while the download ran; a pane showing the same thing now has only its ring moved. **Not run:** the window against this session, an
update's path (no release has shipped a `whats-new.txt`), and the macOS and Linux builds.
- **The onboarding model downloads in the background, said in a status area** (Todd, the same day): "The 'Getting
  Mila Ready' should then have an onboarding 'status' area (like the browser download progress indicator)." Once the
  download begins, every pane after it carries a small indicator of how far it has come, and the person goes on
  through the panes while it runs.
- **A person who reaches the end before it is ready meets a fuller *Getting Mila Ready* screen**: "If after all the
  user has gone through to the end and the onboarding is still not ready we should put up a more detailed 'Getting
  Mila Ready' screen." It replaces round 8's *Getting Ready* pane, and ends when the model is loaded.
- **No model is named or described anywhere in the onboarding** (Todd, the same day): "we do not need to show the
  model details at all. The user should not care. Mila is an 'intelligence' harness. A simple chrome like progress
  indicator in the overall onboarding window is sufficient." The indicator belongs to the onboarding's window, not
  to one pane, and the full *Getting Mila Ready* pane shows only once the other steps are done and it is still
  running.

## 4.4 An onboarding model, and the interview (round 8, decided in part, not built)

Todd, 2026-11-24: the setup gets Mila Assistant running as fast and efficiently as it can, with as few panes as it
can, by a fast install of a universal model; then the model sets the assistant up with the person, "harnessing
intelligence for the user from top to bottom". The words are Todd's of the same day: **the model hub** is where Mila's
models are published (`mila-llm` on HuggingFace), **model metadata** is what a model there says of itself (its
`mila.json`), and **the onboarding model** is the model the setup installs.

**Decided (Todd, 2026-11-24):**

- **The onboarding model works on every compute backend** and is sized for 11 to 12 GB of GPU memory, a Gemma 4 model
  to start. The setup shows no choice of it.
- **Its download begins in the background at once**, at the first launch, before any pane is answered: "There are no
  license restriction on downloading."
- **Mila's terms are accepted before the first chat**, on a pane titled *Terms and Conditions*, replacing round 5's
  *License*.
- **Welcome stays, and says what Mila Assistant is and, briefly, what the setup and the first chat will do.** This
  reverses round 5's removal of the line about what follows. Its words come once the process is settled: "Let's get
  the process right first then the actual prose/language will be easy after that."
- **A model's metadata states that it is onboarding ready**, and there may be several such models: "Why can't the app
  just download one based on the hardware the user has?" So the app picks one for this computer from those on the
  model hub; a new onboarding model needs no release of Mila.
- **The interview is the workspace's first chat**: Mila, on the onboarding model, learns what the person needs and sets
  the assistant up. It is section 1's onboarding from here on; the panes only get a model running.
- **A computer with less memory gets an assistant configured differently**, perhaps chat only. Not yet thought
  through.

**The process, Claude's proposals taken by Todd "to start with" (2026-11-24), to be changed by eye:**

1. **Launch.** In the background the session reads the model hub's list, picks the onboarding model for this computer
   and starts its download. Nothing waits on the person.
2. **Welcome.** What Mila Assistant is and what follows. **Continue**.
3. **Terms and Conditions.** **Accept** once read to the end; **Quit** declines. **Go Back** to the welcome.
4. **Waiting**, only while the download or the load has not finished: their progress, and **Quit**, which keeps what
   arrived. There is no **Cancel**, as there is no way on without the model. A model hub not reached at launch is
   asked again while the welcome and the terms show; *Try again* appears here only when it still fails. Too little free
   disk space is said here, checked before the download starts.
5. **The workspace opens on the first chat**, and the interview runs.
6. **The interview ends** with the assistant set up. A larger model it suggests downloads in the background, and Mila
   changes to it when it is installed, the conversation kept.

- **A person who stops part way** returns where they stopped: accepted terms and an unfinished download open on the
  waiting pane; an unfinished interview reopens the first chat. The session's settings keep the interview's end
  beside the terms' acceptance. *Fixed 2026-11-25, found in Todd's walk:* as built at `+164` a launch part way through
  the interview showed the panes again from the welcome, since the session kept their end only in memory and the page
  began at the welcome. The version kept at the panes' end (`last_version`) now says they were gone through
  (`onboardingPanesGoneThrough`), and the page then shows *Almost Ready* until Mila is ready and opens the interview's
  chat; the mock's scenario 22 plays it.
- **A store that already holds models** (`Mila/Store`, shared with Chat) still installs the onboarding model, so every
  first chat runs on one; a store that holds it downloads nothing.
- **The interview again** is asked for in the person's own words; no button.

**Proposed (Claude's, for Todd):**

- **How the app picks:** the planner plans each onboarding-ready model on this computer, as section 5 plans every
  published one (two small reads of each, before any download), and the pick is the first that runs here, agentic
  here first, then the smaller download, so the first chat comes soonest. The hardware is read through the planner,
  never by the GPU's name or a table of sizes (section 3).
- **The metadata's mark** is `capabilities.onboarding`, beside `agentic`, and like it a measured verdict: written only
  for a model the interview has been measured on. A reader that does not know it ignores it, so an older Mila is
  untouched.
- **The first onboarding model is Gemma 4 12B Q4_0.** It runs on CUDA and on Metal, in the format Google trained it
  for, and is measured agentic. Its weights are 7.1 GB, and its cache 8 KiB a token in the global layers plus a
  constant 335 MB in the sliding ones (`../Quantization.md`), so about 8 GB at 64K and 8.5 GB at 128K (arithmetic),
  inside the budget and inside the M1 Pro's 11.84 GiB working set.
- **The suggestion's card, Install, Skip and *Ready* go.**
- **The load may run before the terms are accepted**, and only a turn waits on them. Today the session refuses
  `model.load` until acceptance (section 4.1); that changes with this.
- **The assistant's features follow the planner and the model metadata, never the hardware's name:** tools, projects
  and the folders Mila reads when the model is agentic here (section 5's meaning), the conversation alone when it is
  not.
- **The interview's settings are the application's own tools** (what to call the person, which folders Mila may
  read, the person's rules for tools), so it adds nothing to the library. A larger model it suggests downloads in the
  background, and the person never waits on it. The interview is measured on the onboarding model before it is
  promised.

**Built, Claude, 2026-11-24 (Todd: "Option (a), go ahead and build it"), on Windows.** Run once through
`mila-session` alone, with a script in the page's place, the local model hub (`MILA_HUB_DIRECTORY`) and a throwaway
Mila folder, on Gemma 4 12B Q4_0: the pick and the download began at 0.2 s, the terms were accepted during it, the
6.96 GB arrived from the local folder by 23 s and the model was loaded by 25 s; with it installed, the interview's
chat opened at 2.1 s, `remember_name` kept "Alex", the second question came, and `finish_setup` ended the
onboarding at 10.2 s. A second launch after quitting at the opening words reopened the same chat. The window was not
run against the session, and the model's two answers are one run, not a measurement. Found in the run and fixed: the
interview's tools waited for the person's approval, so both are allowed without asking.

- **Mila's first message is Mila Assistant's own words** (Todd chose (a) over the model writing it, which the agent
  cannot do: it always answers a message from the person). `Apps/Assistant/Prompts/interview-opening.txt` is the
  chat's first reply, shown at once; the model takes over from the person's answer under
  `Prompts/interview.json`, whose tools are `remember_name` and `finish_setup`. Gemma's chat format was read to take
  a conversation that begins with Mila's message; Llama's and Qwen's were not.
- **The library:** `ModelCapabilities::onboarding`, read and written with the manifest's other capabilities;
  `ExportArtifact --onboarding`; and `Deployment::onboardingModel`, the pick among the planned models.
- **The session:** settings `onboarding` (empty, `"started"`, `"done"`), `interview_conversation` and
  `person_name`. With the onboarding due, a worker thread lists the model hub, picks, and downloads, so the session
  answers the person while it runs; its results reach the session's thread as internal requests the page cannot send.
  The model loads before the terms are accepted, and the interview opens once both are done. `models.refresh` online
  while due is the setup's *Try again*. A name the person gives joins every chat's system prompt.
- **The page** reads `session.state.onboarding`, `models.list.onboarding` and the `interview` event; the mock's
  scenario 21 plays it.
- **Not built:** which folders Mila may read and what the person uses Mila for, since no setting holds either across
  chats yet (a folder is a chat's or a project's); the smaller computer's model; Gemma's terms at the setup; free disk
  space; and the macOS and Linux builds, which are not yet run.

**Open:**

- The smaller computer's onboarding model, chosen by measurement; none measured today qualifies.
- Gemma's terms: where the person meets them, now that they did not choose Gemma (question 16).
- Contacting the model hub at launch, before any press: round 5 already asks as *Install Model* shows; this moves it
  earlier, and section 4.2's promise is to be said accordingly.

### The interview's goals (round 10, 2026-11-25)

Set out with Todd while he walked the interview as built at `+164`. The interview is *Setting up Mila*, the
workspace's first chat.

**Decided (Todd, 2026-11-25):**

- **It has two goals:** Mila and the person get to know each other, and Mila is set up on a model suited to the
  person's ongoing chats. "The other 'Setting up Mila' objective is to deploy a more suitable intelligence model for
  ongoing sessions."
- ~~**It ends in a decision:** "Setting up Mila must result in a decision to stay with the current onboarding model or
  move to a new regular session model."~~ *Superseded the same day by round 12: the person is offered no choice of
  model; Mila assesses the need and asks only to download.*

**Proposed (Claude's, for Todd), from the vision's promises (`../Direction/MilaVision.md` 3 and 6):**

- **What is asked passes three tests:** Mila does something differently with the answer at once; the computer cannot
  tell Mila itself; and the person can see where it is kept, and change or delete it. The interview stays a
  conversation of a few turns.
- **What to ask:** what to call the person (built; a full name offered on the way is kept too, for what Mila signs);
  how they like answers, short or thorough, plain or formal; and what they mostly want help with, in the vision's
  areas (section 5: work, their files, organization, research, mail and messages, getting things done). The last
  earns its place as an input to the model's decision, not as a survey.
- **What is read, never asked:** the date, the time and time zone, the language and the region's formats, from the
  operating system on Windows, macOS and Linux, following the person when a laptop travels. None of it reaches the
  model today on Gemma or Qwen (`Issues/Untriaged.md`, 2026-11-25), which comes before any question.
- **Location:** no precise location at the setup. Each platform asks the person for it through its own prompt, and it
  is among the most sensitive facts on the machine. A home area the person offers in words is kept; otherwise Mila asks
  when a request first needs it ("Which city?") and offers to remember it. A search that carries it is an outward
  action like any other (the vision's section 3).
- **Mila says who it is:** what it can do on this computer (tools and projects, or the conversation alone), that the
  person can see and change what it knows of them, and that saying "set me up again" runs the interview again.
- **The model's decision:** the planner decides what runs here and the model metadata what each candidate does
  (section 3); the person's answers say which strengths matter -- measured agentic here for work with files and tools,
  the longest context for long documents, a smaller and faster model for conversation. Staying is recommended when the
  planner finds nothing better here. The choice is put without naming a model (round 9), with what the person gains
  and the download's size; an agreed download runs in the background with the window's ring, and the change of model
  comes between turns with one line in the chat.
- **What is kept:** one record of what the person told Mila, each entry pointing to the chat it was said in, given to
  every chat and shown in Settings, where it can be changed or deleted -- the first piece of the vision's Personal
  Context (section 6), and the job a person's entry in the window could return for (`AssistantInterface.md` 4.1).

**In the way today:**

1. **A change of model clears the conversation** in Mila Assistant (`MilaAssistant.md` 7), so the interview cannot
   carry on after the move. The library carries a conversation across models; only the app clears it
   (`../Direction.md` 4.3).
2. **A conversation crossing families is untested**, and `../Direction.md` 4.3 asks for that test before anything
   relies on it. The interview begins with Mila's message, and only Gemma's chat format was read to take that.
3. **Nothing holds the answers but the name.** The interview asks what the person uses Mila for, acknowledges it and
   keeps nothing.
4. **Free disk space is not checked**, and two models -- about 7 GB and a larger one -- make it matter.

**Open:** whether the onboarding model stays installed after a move (it runs "set me up again", and every first chat
runs on one by round 8); and whether the move happens at once or at the next new chat.

### The first walk, and the interview's next design (round 11, 2026-11-25)

**What happened** (Todd's run on Gemma 4 12B Q4_0, `+164`, one run, not a measurement). The name was kept. Asked what
he expected to use Mila for, Todd asked back *"What are some of the things that you can do?"*; the model read that as
starting on something else, as the prompt's way out allowed, called `finish_setup`, and answered *"You are all set
up, Todd! I can help you with ... organizing your schedule, answering questions, drafting emails, or helping with
creative projects."* The setup was marked done with nothing set up but the name, no model decision put, and two of
four things offered that Mila does not have.

**Why, most important first:**

1. **The order was backwards.** Nobody can say what they will use Mila for before knowing what it can do, so the
   question back was the natural answer. Mila says what it can do here before asking.
2. **Mila had no facts about itself,** so asked, the model gave a generic assistant's list. The interview's prompt
   gave it two tools and nothing of what Mila can do, and *"do not describe how you work inside"* pushed it to vagueness.
3. **The way out was far too wide:** *"to start on something else at once, call finish_setup"* made any change of
   subject the setup's end. A question about the question is not a skip.
4. **The model alone ended the setup.** `finish_setup` worked whenever called, and no code checked what was done, so
   round 10's rule -- the setup ends in the person's decision to stay or move -- held only if the model behaved, against
   the vision's invariant 4.
5. **The tool's result told the model to say something untrue:** *"Tell the user they are set up"*, whatever was kept.
6. **The uses answer had nowhere to go** even had it been given (round 10, *In the way today* 3).

**The next design (Todd, 2026-11-25: "write it into 4.4 and start on the prompt"):**

- **Order:** the greeting and the name; what Mila can do here; what the person will mostly use it for; how they like
  answers, once there is a record to keep it in; then the model decision, which ends the setup.
- **Facts, not recall:** the app gives the model what Mila can do here as data -- the tools a chat gets on this
  install (a search key set, commands on) and the onboarding model's measured capabilities, the facts *What Mila Can
  Do* shows -- and the model says only what that list says. Until the app hands them over, the prompt carries the list
  the app offers every chat today, without the web search and commands that depend on settings.
- **Questions are answered and the interview goes on;** only a plain wish to skip the setup or to start now ends it,
  and a skip is a choice to stay, said as such.
- **The end is the person's decision, recorded in code:** the stay-or-move choice is put to them as a project
  suggestion is put today, and the setup ends when they answer. The model may propose it and cannot declare the setup
  done.
- **The ending says what was kept,** never "set up".
- **Measured before promised:** scripted people on Gemma 4 12B Q4_0 -- one who answers everything, one who asks back,
  one who skips, one who starts on a real task -- each counted for staying on course.

**The steps are the app's, and an unfinished setup is flagged** (Todd, 2026-11-25: "A session should be flagged with
outstanding items?"; proposed, Claude's):

- **The app keeps the setup's steps:** the name, what Mila can do (said), what the person will mostly use Mila for,
  and the stay-or-move decision, each *done*, *skipped* or *outstanding*. A skip is an answer, so it is not outstanding.
  The interview's tools mark them as they happen, so what is left is known in code, never from the model's words.
- **They are what the setup has to accomplish, not a plan, and nothing of them is drawn in the chat** (Todd, 2026-11-25,
  on seeing them built as the chat's plan, with its panel: "Why did you add a 'plan'? I do not like this shown at all.
  This needs to be what the 'setup' needs to accomplish.").
- **An unfinished setup is under *Needs you*** on the overview (4.3 of `AssistantInterface.md`): *Setting up Mila*,
  what is still to do, and *Continue*.
- **Only the decision is required;** the name and the uses may be skipped (proposed: round 10 made the decision the
  setup's end, and nothing else is).
- **At a launch** with the setup unfinished, the workspace opens with the *Needs you* card rather than on the chat
  (proposed: the chat today reopens at every launch, which keeps the person in the setup until it ends).

**One setup chat, which ends and hands over to a new chat** (Todd, 2026-11-25: "once finished ... should the model
start a new chat or say that it is ready for a new session? What is the handoff"; Claude's answer, taken: "write it
all into 4.4"). *Its close and the handoff are superseded by round 13: the setup chat goes on as the person's first.*

- **One per workspace.** "Set me up again" carries on in it, so there is never a second *Setting up Mila*.
- **The app ends the setup,** once the decision is made and every other step is done or skipped; the model cannot.
- **Mila's last reply there says what was kept and decided,** in a sentence or two, and that a new chat is where to
  start; "set up" is true by then.
- **The setup chat then closes:** its steps all ticked, no message box, *New chat* in its place; the workspace's own
  box starts one too. It stays as the record of what was decided.
- **The first chat of the person's own is a new one,** on the chosen model once it is ready and on the onboarding
  model until then. So the move happens at the next new chat, and no conversation crosses models at the handoff --
  round 10's *In the way today* 1 and 2 no longer stand in the setup's path, though reopening an older chat on a new
  model still needs the test across families (`../Direction.md` 4.3).
- **Not taken:** Mila opening a new chat by itself with *What would you like to start with?*, which would be fixed
  text again, since the model cannot speak first, and would leave an empty chat behind each time someone only looks.

**Built, Claude, 2026-11-25 (Todd: "Let's build the setup steps and the handoff"), on Windows:**

- **The objectives** are `Assistant.SetupSteps` -- *What to call you*, *What you will use Mila for*, *Which model Mila
  runs on*, each pending, done or skipped, with what was said -- kept with the setup's chat under `setup`, never as
  its plan, so no plan panel is drawn. The chat's summary carries what is still outstanding (`setup`), and the overview
  puts an unfinished setup under *Needs you*: *Still to do: what you will use Mila for and which model Mila runs on*,
  with *Continue*. Built first as the chat's plan and changed the same day at Todd's word above.
- **One question about the person** (Todd, 2026-11-27, replacing *What you will use Mila for*): in all three of his
  HelloWorld walks he answered "what do you expect to use me for most?" with who he is ("I am a software engineer that
  likes to code in c++"), kept as uses and said to every chat as "will mostly use you for: Software Engineer who uses
  C++". So Mila asks *a bit about yourself and what you would like my help with*, the step is *About you, and what you
  would like help with*, and `note_uses` takes `about` and `uses`, either or both, kept as `person_about` and
  `person_uses`. Every chat's prompt says each on its own line. Settings has an *About you* area with the name and
  both, which the person can change or clear: what every chat is given about them is theirs to see. *A full name
  keeps the first* (the same day): told "my name is Todd Thomson", the 26B kept and used the whole; Mila now keeps the
  first name and says the person can ask to be called otherwise.
- **The tools:** `remember_name` and `note_uses` mark their steps (`person_uses` and `person_about` join every chat's
  system prompt, as the name does); `skip_question` skips the name or the uses, or with `setup` the whole setup, which stays on the model
  and ends it; `offer_model_choice` is refused until the name and the uses are each done or skipped, and puts the choice
  as an offer of kind `model`. `finish_setup` is gone.
- **The choice** is built by code: the planner's suggestion from the model hub's list, offered as a move only for what
  it gives over the model Mila runs on -- measured agentic here, or a longer context -- with its download's size and
  no model named; otherwise *This model suits this computer best.* It is answered in words (a bare yes or no, read by
  app code as other offers are) or with its buttons (*Move to it* and *Stay*, or *Stay on this model*), through
  `suggestion.decide`.
- **The end:** the answer marks the model's step, ends the setup and adds the app's own line -- *I'm set up: I'll call
  you Alex, and I'll stay on the model I run on now. Start a new chat whenever you're ready.* -- so the last line is
  true whatever the model would say. The setup's chat then takes no more messages (the session refuses them), and the
  page puts *New chat* where its message box was; `session.state` carries `setup_conversation`. The setup chat is kept
  (`interview_conversation` is no longer cleared at its end).
- **A move** keeps the model in `next_model` and downloads it on the worker, as the onboarding model's download runs,
  so the person goes on meanwhile; the first new chat once it is installed loads it (`conversation.new`), as does a
  launch, and a download a quit stopped goes on at the next launch.
- **Checked:** the session's tests (`SetupSteps`, the summary, the settings) and the page's (scenario 21 played through
  the page: the objectives kept and no plan drawn, *Needs you* with what is still to do, the choice answered in words
  and with its button, the chat closed and refusing a message); the window over the mock, both widths, light and dark,
  *Continue* on the card reopening the chat. **Not run:** the real session's interview
  past the name (Todd's walk), a move (the local model hub holds the onboarding model alone), a skip of the whole setup,
  and the macOS and Linux builds.
- **Not built:** the launch with an unfinished setup still reopens its chat rather than showing the *Needs you* card
  (proposed above, Todd's call); "set me up again"; the facts from the app beyond the tools; the record shown in
  Settings; the measurement.

*Built the same day, the prompt's part (points 1, 3 and 5, with the list in the prompt):* `Prompts/interview.json`
says what Mila can do before asking what the person will use it for, answers a question and returns to the interview,
ends only on a plain wish to skip or to start; `finish_setup`'s result says what was kept. Its list is kept by the
tools a chat on this computer is offered (a folder on Windows and macOS, a search key, commands), not the interview's
own. From Todd's next walk (*"when someone introduces themselves the correct response should contain a 'hi' back"*):
Mila answers what the person said before asking anything, greets them back by name, and keeps the name they
introduce themselves with rather than their full name. Not built: the facts from the app, the decision in code, the
steps and the flag, the handoff, the record, the measurement.

### No choice of model: Mila assesses, the person agrees to a download (round 12, 2026-11-25)

*Superseded by round 13 (below): one download, and no assessment after it. Kept as the record of what was tried.*

**Decided (Todd, 2026-11-25),** on seeing the choice of model built (round 11): "The model is nothing the user should
focus on. Mila is an intelligence harness. The user does not need to be offered the choice of a model." Then: "Mila
will assess the need for a suitable model from the user interview process. If the user does not accept the download
and install then the onboarding model will then stay. Mila must be able to later (through normal usage) need to
remember this and perhaps offer again if it fails at tasks, etc."

- **The person is offered no choice of model.** The setup is about the person; which model runs is Mila's to decide.
- **Mila assesses the need from the interview:** what the person said they will use Mila for, against what the
  onboarding model does on this computer and what a published model would do better (the planner, section 3).
- **When something more suits them, Mila asks only to download and install it,** in what it gives the person and the
  space it takes, no model named (round 9's rule). Accepted, it downloads in the background and the next new chat runs
  on it (round 11's mechanism, kept). Declined, the onboarding model stays as the person's regular one.
- **A decline is remembered,** and Mila may offer again later, from normal use, when it falls short at a task.
- **A choice of model exists only in Settings, as an advanced setting,** through the library's deployment API. The
  Settings review is still to come and will change much there.
- **Mila Assistant is not a showcase of the library's API.** It is "more of the old Mila::AI idea": a harness between
  the person and the library that makes these decisions on the person's behalf.

This supersedes round 10's decision that the setup ends in a choice to stay or move, and round 11's `offer_model_choice`
and its card (*Move to it*, *Stay*, *This model suits this computer best*).

**Proposed (Claude's, for Todd), built below:**

- **The person's objectives** are what to call them and what they will mostly use Mila for; the third, Mila's own, is
  *What Mila needs for your work*, done once Mila has assessed it and, when it asks, the person has answered.
- **The assessment is code, from the model's reading of the answer:** `note_uses` also takes what the uses need, as
  the model reads them -- conversation, work of several steps with files and tools, long documents -- and
  `assess_needs` (in `offer_model_choice`'s place) compares the planner's suggestion with the onboarding model on
  those needs alone: measured agentic here for files and tools, a longer context for long documents.
- **Nothing needed, nothing asked:** the setup ends there, and Mila says it is set up.
- **Something needed:** a card, *To do more of this for you -- work of several steps with your files -- Mila needs to
  download about 15.2 GB*, with *Download* and *Not now*, answered by its buttons or a bare yes or no. Either answer
  ends the setup with the app's line.
- **The decline is kept in the settings** (what was declined, and when). Offering again from use is not built now:
  which failures count is a design of its own, measured before it is promised.
- **The harness's decisions stay in the Assistant session for 0.21**, behind one interface written as the harness's
  own, and move into a library layer when 0.22's family-blind harness is designed.

**Built, Claude, 2026-11-25, while Todd was away ("Please continue on where you can"), as proposed above,** on Windows:

- **The objectives** are *What to call you*, *What you will use Mila for* and *What Mila needs for your work*
  (`Assistant.SetupSteps`). `note_uses` takes `needs` (`conversation`, `files_and_tools`, `long_documents`), kept as
  `person_needs`; `assess_needs` replaces `offer_model_choice` and its card. With nothing needed it ends the setup
  itself; with something needed it offers a download of kind `download` (`download`: `name`, `size_bytes`, `gains`),
  the card reading *To be better at work of several steps with your files and tools, Mila needs to download about
  14.4 GB*, with *Download* and *Not now*. `Download` keeps `next_model` and starts it on the worker; *Not now* keeps
  `declined_model` and `declined_at`. Either ends the setup with the app's line: *I'm set up: I'll call you Alex, and
  I'm downloading what I need for your work, about 14.4 GB; your next new chat uses it*, or *... and I'll carry on as I
  am*. A skip of the whole setup downloads nothing.
- **Run on the real session** (Gemma 4 12B Q4_0, the local model hub holding the 26B-A4B and Qwen `fp4-cb3`, a
  throwaway Mila folder, the 16 GB card; six runs answering "coding ... multi-step plans in my project folders"):
  every run greeted the person by name, read the needs as `files_and_tools`, and was offered the 26B-A4B's download,
  14.4 GB, no model named. Accepted, the download ran in the background (about 46 s from the local hub) and the next
  new chat loaded the 26B-A4B with the FP8 cache at 73,728 tokens; declined in words, the setup ended and the decline
  was kept. One run in six left out the question about uses, and two of six the question about the download, before
  the prompt asked every setup reply to end with its question; three runs after, every reply did. One reply called the
  download "small", and one said Mila could "manage your files". Runs, not a measurement.
- **Found in the runs and fixed:** the offer was dropped, as `assess_needs` runs without a question and so had no call
  announced to keep it on, and a bare *yes* went to the model, which said it had started the download when nothing
  had (the call is now announced first, as `read_project_notes` is; the prompt says the app downloads); a background
  download made the session busy, so the person could not type until it finished (it no longer does, while the
  onboarding model's still does); *No thanks, not now.* was no answer, being two refusals together (answers of one
  kind together are now that answer, for every offer, `Assistant.OfferAnswer`); and pricing a load on the whole card
  priced the 26B-A4B uncompressed at 95,232 where its load got 37,888 (a change of model now releases the resident one
  and prices on the card as it is free; a `log` event says what each cache priced at).
- **Not built:** offering a declined download again from use; the facts from the app beyond the tools; the setup in
  the real window (the mock's scenarios 21 and 23 play it in the page); the macOS and Linux builds.

**The models offered for coding and multi-step work, and the cache they run with (Todd, 2026-11-25):** "I'd like the
Gemma 4 26B A4b 4 bit quantized model and the Qwen 3.8 27B model avail in the local store ... They should be avail for
coding/programs and multi-step iterative plans and processes ... They would only target a CUDA 16GB compute env." Then,
of the KV cache: "Yes, use the FP8 cache for those models."

- **On the local model hub** (`Data/Hub`, for testing the setup on this PC): Gemma 4 26B-A4B Q4_0, packaged from the
  store's copy with its display name, licence and the `agentic` mark it carried; and Qwen 3.8 27B `fp4-cb3`, the build
  `../Qwen3.8.md` chose for agentic work on the 16 GB card (FP4 plans 16K there, and the 2.82-bit build drops closing
  brackets), packaged from `Data/Models/Qwen` with the publisher's sampling. The current library reads both.
- **Qwen 3.8 27B carries no `agentic` mark:** `../ModelEval.md` Phase 8, which would award it, has not run, and a mark
  is a measured verdict (section 3). Until it does, the assessment offers it for nothing that needs one.
- **The cache:** the Assistant asked every model for the uncompressed cache (since `+105`, with no reason recorded), so
  on the 16 GB card the 26B-A4B planned 37,888 tokens and `fp4-cb3` 32,768, short of the 64K agentic work is measured
  at, where each was measured at 65,536 with the FP8 cache. The harness now prices both and takes the FP8 cache only
  where the uncompressed one falls short of `kAgenticContextLength` and FP8 reaches it (`Assistant.KvCacheChoice`):
  for a load, priced on the card as it is free once the resident model is released (pricing on the whole card was
  tried and misjudged it, above); for the published list, each model planned
  both ways and the library's ranking applied to the result (`rankPublishedModels`, exported for it). A change of
  context keeps the cache the model was loaded with. Every model that reaches 64K uncompressed is unchanged.

### One download, of the model that suits this computer (round 13, 2026-11-26)

**What happened** (Todd's walk at `+165`, the 16 GB card, the local model hub; one run, not a measurement). The
interview on Gemma 4 12B Q4_0 did what its prompt asked: it kept *Todd*, said what Mila can do, read *"all features
and capabilities"* as all three needs, and put *"a download that takes about 14.4 GB"*, no model named. Todd said yes;
the app's line ended the setup and the chat closed. Todd, the next morning: the words -- "What is it downloading?
Users will understand model. Mila has chosen a new, larger model." -- no progress of the download in the chat, a new
chat pressed before it finished would have run on the 12B without a word, and "the chat just ended abruptly". The
first chat on the 26B-A4B, asked *"Are you ready with the new model?"*, searched his chats for *new model*, found
nothing, as the interview never said *model*, and asked whether he meant a machine learning model
(`Issues/Untriaged.md`). And the two-step's gain is small where the larger model fits: at Todd's 10 MB/s the 12B's
6.96 GB is about 12 minutes, the 26B-A4B's 14.41 GB about 24, and both together about 36 (arithmetic), so the first
chat came 12 minutes sooner and the person waited 12 more in all, with their first conversation on the weaker model.

**Decided (Todd, 2026-11-26: "Agreed, file it in Future.md and go to one download"):**

- **One download.** The setup downloads the model that suits this computer best, the planner's suggestion (section 5,
  the library's ranking: agentic here first, then the larger download), and the interview runs on it. There is no
  onboarding model, no second download and no handoff between models. This supersedes round 8's onboarding model and
  round 12's assessment and its download.
- **People understand *model*.** The setup says that Mila downloads a model chosen for this computer, and its size.
  This revises round 9's *no model named anywhere* for the words: the model is spoken of; its name stays out
  (proposed, Claude's: Todd did not say whether the name may appear, so it does not until he does).
- **Starting on a smaller model while a very large one downloads** is `Issues/Future.md`'s, triggered by the
  expected wait rather than the graphics memory, and taken up when Mila publishes a model whose download makes it pay.

**Proposed (Claude's, taken with the decision):**

- **The setup's steps are the person's two:** what to call them and what they will mostly use Mila for, each done or
  skipped, and the setup ends when both are. `assess_needs`, the offer of kind `download`, `next_model`,
  `declined_model`, `declined_at` and `person_needs` go; `note_uses` takes the uses alone.
- **The setup chat carries on as the person's first chat.** No line from the app, no closed chat and no *New chat* in
  its place: the tool that ends the setup tells the model so, and Mila, in its own words, says what it kept and asks
  what they would like to start with. From the next message the chat runs as every chat does, on the general prompt
  and its tools. This supersedes round 11's *The setup chat then closes*, which existed only because the next model
  was not the one the setup ran on.
- **Its title follows** (Todd, 2026-11-26, after his walk, where a project gathered the setup's chat and its path read
  *Workspace / HelloWorld / Setting up Mila*): the chat is *Setting up Mila* until its first message after the setup,
  which names it as a new chat's first message does. The setup stays on record in the chat's stored steps.

**Built, Claude, 2026-11-26, on Windows.** As above. Run on the real session (the local model hub, a throwaway Mila
folder, the 16 GB card): the setup picked the 26B-A4B, downloaded and loaded it in about 55 s at 75,776 tokens, the
interview ran on it, Mila closed the setup in its own words, and the next message in the same chat called
`request_folder`, offered by the general prompt alone. Todd walked it in the window the same day. **Not run:** the macOS
and Linux builds.

**Open:**

- **The macOS part.** The planner suggests by what fits and by the `agentic` mark, and neither says whether this
  backend runs the model whole. On Metal in v0.21.0 the Gemma 26B-A4B and Qwen 3.8 whole are `Issues/Vnext.md`'s
  (`../Metal/MetalBackend.md`), so a Mac with the memory for either could be suggested a model Metal does not yet run;
  round 8's pick was held to the onboarding-ready models, of which Gemma 4 12B runs on Metal. The one download is not
  done on macOS until the suggestion leaves out what the backend cannot run (put to Todd).
- **The library's onboarding pick** (`Deployment::onboardingModel`) and the `capabilities.onboarding` mark it reads
  (`ExportArtifact --onboarding`) are no longer read by the app; removing them is a library change put to Todd.

## 5. The suggestion, as built at +142

**Shape A, fetch what planning reads.** Of round 1's three shapes, A is built: it plans the very bytes the install
will download, and needs no manifest change. B would copy into the manifest what the weights already carry; C plans
on Mila's own machine before the model goes on the model hub, which section 3's first rule rules out.

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
  `Assistant.Models.ixx` did by hand), `downloadBytes()`, and `weightsHeader( hub, store )` -- two small reads, the
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
4. *One suggestion or a ranked few.* One, with *Choose another model* (round 7: **Skip**, and Settings' *Models* for
   another).
5. *When Mila suggests.* On the first run, and as a mark in Settings' *Models* whenever the published list is shown.
   No alert when a model is published.
6. *Whether the first run loads after the install.* It loads, as the session already did.
7. *Two GPUs.* The planner plans every allowed device and its best plan is the answer; the app chooses nothing.
   Not run on a two-GPU machine.
8. *The download.* Its size before it starts. Free disk space is not checked first.
9. *Per activity or overall.* One overall. Activities are gone since `AssistantInterface.md` phase 3.
10. *Before the suggestion.* *Welcome to Mila* and one line. Round 3 replaces it: a welcome, then the licence
    (section 4.1).
11. *None fits, or no model hub.* Section 4, step 6.
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
18. *A Conclusion pane.* The guidelines end on one; Todd's three steps do not. Proposed: add it, since today the
    overview follows the load with nothing said about where the model is changed.
19. *The 2002 window rules.* The reduced menu bar: yes, on a packaged Mac. Windows and Linux packages have no menu
    bar at all (`Window/src/main.js`), so there is nothing to reduce. The dimmed close and zoom buttons: no, since a
    person who closes the window has declined, which quitting already covers, and Windows has no dimmed close.
20. *The model pane when the model hub cannot be reached, or nothing runs here.* Section 4 step 6's screens, with
    **Go Back** to the licence; nothing past the model pane opens until a model is installed, and the window with
    *Install a model to start* is no longer reached from the first run.

**Round 4's questions** (Todd, 2026-11-18, "Proceed with your recommendations"):

21. *Contacting the model hub.* Section 4.2 is the step, with its one more press on the first run.
22. *The welcome's byline.* *Local LLMs on hardware you own -- at the metal* speaks to people who already know what an LLM
    is, and the vision's first-run reader may not. It is to change so that it says what Mila is for that person. **The
    words are Todd's and not yet chosen,** so the line stands until then. Any line promising privacy waits until the
    tests of `AssistantInterface.md` 5.7 pass.

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

- Choosing a model per conversation after the first run (`MilaAssistant.md` 3.2's library query, a proposal for
  `Mila::Agents`).
- The screens for a store with models and none loaded (`MilaAssistant.md` 3.4).
- The renaming of the app and its folder to Mila Assistant.
