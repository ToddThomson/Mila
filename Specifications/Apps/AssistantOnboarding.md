# Mila Assistant Onboarding

**Status:** design, round 2 (2026-11-13). Round 1 set out the facts and twelve questions. Round 2 is a one-shot
build at `0.21.0-dev+142`, which Todd asked for to see how Claude would design and build it ("test out how you would
design and implement this in a one shot attempt", Mac first). Every answer in section 6 is Claude's provisional one,
marked as such, until Todd settles it.

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

1. With nothing installed and nothing loading, the empty conversation shows *Welcome to Mila* and one line: Mila
   runs a language model on this computer, so what you ask stays here; to start, install one. The message box and
   the header's model menu are hidden, as on every screen that asks for a model (`+141`).
2. The screen asks the session for the published list as it first shows (`models.refresh`, online) and says
   *Finding the model that suits this computer* until the list comes back.
3. The suggestion, in a card: *Suggested for this computer*, its name, what it can do (*Works through tasks with
   tools* only when it is agentic here, section 5; else *Answers questions and holds a conversation*; then whether
   it thinks and what it reads besides text), its download and the conversation length it holds on this computer,
   and its licence by name. One *Install*.
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
10. *Before the suggestion.* *Welcome to Mila* and one line.
11. *None fits, or no publisher.* Section 4, step 6.
12. *The licence.* Shown by name. No agreement wording: what installing means for the licence is Todd's.

**Open, found in the build:**

- **Planning with a model resident** uses the request's whole-card form (`withWholeDeviceMemory`, `Deployment.md`
  12.5), as a listing does, so the resident model's memory does not count against the others. A shortfall then
  still quotes the free memory in its sentence; the wording for the whole-card case is not written.
- **The time to plan every published model is not measured.** Each costs two small reads of its weights the first
  time and a graph construction on the device every time.
- **Disk space** before an install (question 8).

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
