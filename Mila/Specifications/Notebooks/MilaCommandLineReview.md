# Review of the `mila` Command Line Utility

**Status:** Review, opened 2026-10-10 at `0.21.0-dev+57`; scope decided 2026-10-11 (section 4). The utility is
`mila-store`, a model-store tool and nothing else; MIS and Chat are applications that need no front door. **Built
at `0.21.0-dev+58`** (section 6): the move to `Mila/Apps/`, the rename, `serve` removed, and every verb of section 4
except `show` and the fit column. **Built at `0.21.0-dev+62`** (section 7): MIS's own `models` and `install`.
**Open:** the rest of section 5.

**Why now.** Todd asked, while settling how the inference server is published (`BACKLOG.md`, Applications Ship),
whether `mila serve` is the right way to start it, and asked for the utility to be reviewed for v0.21.0.

---

## 1. What exists

`Mila/Tools/Cli` builds the `mila` executable: `Cli.ixx` (393 lines) and `main.cpp`, which only resolves the
executable's own directory. Three verbs:

| Verb | Does | Through |
|---|---|---|
| `mila install <name>...` | Installs published models into the store, with progress and the licence identifier | `ModelResolver::pull` over the default hub, in process |
| `mila models [--online]` | Lists what is installed, or what the hub publishes with a Mila manifest | `ModelStore::list`, `IModelHub::listModels` |
| `mila serve [args]` | Starts the inference server | `std::system` on `mila-server` in a virtual environment beside the executable, or `MILA_VENV_DIR` |

There is no `chat` verb, by a decision of 2026-08-15: `mila-chat` is a binary beside `mila` and a verb would hide
nothing, while `mila-server` is a console script inside a virtual environment, which a verb does hide. The same
decision keeps the store tool free of CUDA, so `mila install` works in a CPU-only build, on a CI runner and in an
image build layer.

The half that calls the library has had no defect found in it. Every defect found so far was in the forwarding
half.

## 2. Findings

1. **The command never reaches the main Python install path.** `mila` ships in the Docker runtime image
   (`Docker/Dockerfile.runtime:175`, linked into `/usr/local/bin`) and in a source build's output, and nowhere
   else. The website's Get Started tells the reader to run `mila install Llama-3.2-3B-Instruct-fp4`
   (`Web/content/start.md:50`). A user who ran `pip install mila-llm` has no `mila` command; the store is reachable
   from Python only, as `mila.ModelStore`.
2. **`mila serve` works only in the container's layout.** It looks for `venv/` beside the executable
   (`Cli.ixx:54`) or `MILA_VENV_DIR`. From a source build on Windows the server's environment is
   `Mila/Adaptors/Inference/Server/.venv`, so `mila serve` answers "'mila-server.exe' is not part of this build".
   Once the server is on PyPI it is a console script on `PATH`, which the lookup never tries.
3. **`serve [args]` forwards arguments the server does not read.** `mila-server` is configured by `MILA_*`
   environment variables alone (`app.py` `main`, "takes no arguments"), so whatever follows `serve` is dropped
   without a word.
4. **The store can do more than the utility exposes.** The library's store has `remove` and `usage` (blob,
   reclaimable and partial bytes); the binding projects both. The utility has neither: a user can install a 16 GB
   model from the command line and cannot remove it there. `mila models --anything` lists installed models rather
   than refusing the flag (`Cli.ixx:353`).
5. **It lives where nothing ships.** `Mila/Tools/README.md`: "None of it ships." The utility is in the image and on
   the website, so it is a published surface kept in the developer tree. v0.22.0 moves Chat and the server to
   `Mila/Applications/`; where the utility moves is not recorded. (Answered in section 4.1: `Mila/Apps/Tools/Store`,
   in v0.21.0.)
6. **A naming decision is on record that this review's question contradicts.** `Mila/Issues/Future.md`, "`ExportArtifact`
   needs a name, subcommands, and its store verbs handed back to `mila`", records names decided 2026-09-27: this
   tool `mila-cli`, ExportArtifact `mila-package`, Chat `MilaChat.exe`. Todd's question of 2026-10-10 uses `mila`
   and `mila serve`, as the executable is named today.
7. **Two tools install into the store.** `ExportArtifact --install` installs a local package; `mila install` pulls a
   published one. Chat (`Chat.ModelCatalog.ixx:387`) and the server (`model_worker.py:90`) tell a user without a
   model to use one or the other (the same `Future.md` entry).

## 3. How a pip user would get `mila`

**Reopened by section 4.** Written 2026-10-10, when the utility still carried `serve`. With `serve` gone and MIS an
application, whether a pip user needs this tool at all turns on section 5's first question; the two shapes below
stand as they were written.

Only if section 4 keeps the store verbs in this utility. Two shapes:

- **(a) The native executable inside the `mila-llm` wheel**, installed as a script, so `pip install mila-llm` gives
  `mila install` and `mila models`, and `mila serve` runs `mila-server` from `PATH`, or names
  `pip install mila-llm-server` when it is missing. One implementation everywhere: the image, the wheel and a
  source build run the same C++ `mila`. Cost: the wheel stages a second binary, which finds the CUDA libraries the
  way the extension does, and the release script builds and checks it.
- **(b) A Python `mila` console script in the wheel**, over `mila.ModelStore`. Simple to package; a second
  implementation of the same commands beside the C++ one in the image, and the two drift.

**Recommendation: (a)**, with findings 2 to 4 fixed in the same work — `serve` tries `PATH` first and takes no
arguments, and `remove` and `usage` are added.

## 4. What the utility encompasses

**Decided 2026-10-10 (Todd): a published utility is for the end user only.** Whatever ships carries the verbs a
person running Mila needs, and nothing a developer of Mila or a model producer uses.

- **Packaging — ExportArtifact's verbs. Out:** packaging, validating and transcoding are a model producer's work,
  which is the `mila-package` tool of finding 6.
- **Publishing. Out:** `Tools/Publishing/publish_model.py` uploads a package, which is a producer's act; the library
  itself never uploads.
- **Chat. Out**, settled 2026-08-15 and reread against this scope: unchanged.

**Decided 2026-10-11 (Todd): MIS is an application, and the utility fronts no application.** The adaptor concept
becomes Apps, in v0.21.0 (section 4.1). As applications, MIS and Chat each carry their own
install, command and documentation, and Mila documents how to use each. The binding stays consumer-blind
(`CLAUDE.md`): no extra, script or import in `mila-llm` reaches MIS.

- **`serve` is dropped.** The reason that kept `chat` out in August now holds for the server: once
  `mila-llm-server` installs from PyPI, `mila-server` is already a command, and a verb hides nothing. Findings 2 and
  3 close with it, and so does `Vnext.md`'s "`mila serve <args>` loses every argument on Windows".
- **The vLLM shape belongs to MIS.** `vllm serve <model>` is one Python program serving in process; the native
  `mila` launching a Python server was where every defect in this utility was found. The same ergonomics come from
  MIS itself — `mila-server <model>` — which today takes the model from `MILA_MODEL` only (`app.py` `main`). Windows
  is not an obstacle: the binding and MIS already publish and pass the clean room there.
- **Installing MIS** is the application's own matter: `uv tool install mila-llm-server` (uv supplies a matching
  Python and puts `mila-server` on `PATH`), a virtual environment with pip, or the image's `serve` verb. On a Mac
  MIS runs nowhere today — no macOS wheel, and no family has a CPU or Metal path — but a Mac client can use a MIS
  on another machine.

**Decided 2026-10-11 (Todd): the utility is `mila-store`.** It replaces the `mila-cli` of 2026-09-27 (finding 6) and
sits beside `mila-package` and `mila-server`. It stays the native C++ binary. Its verbs, each exposing what
`Mila::Distribution` already has:

| Verb | Library | Today |
|---|---|---|
| `models [--online]` | `ModelStore::list`, `IModelHub::listModels`, `planDeployment` | exists, without the fit verdict |
| `show <name>` | `locate` / `describe`, manifest, `planDeployment` | — |
| `install <name\|dir>` | `ModelResolver::pull`, `ModelStore::install` | published only |
| `remove <name>` | `ModelStore::remove` | — |
| `usage [--clean]` | `ModelStore::usage`, `ModelStore::clean` | — |
| `--version` | `getAPIVersion` | — |

- **The fit verdict is in.** `models` gives each model's verdict on this machine, as Chat's `/model list` does, and
  `show` the largest context that fits per device. The planner reads devices through the registry and names no
  backend, so the tool carries no CUDA assumption: whatever devices are present are planned against. On a machine
  without a CUDA device the answer today is that nothing fits — the CPU device reports no memory (`Untriaged.md`,
  filed 2026-10-11) and no family has a CPU path. The tool does not change when either gap closes.
- **`install <dir>`** answers finding 7 for an end user handed a package. `ExportArtifact --install` stays a
  producer's verb in `mila-package`.
- **`rename` is out.** The library has it; an end user does not need it, since an install from a directory names
  the model.
- **A device listing is out.** It is not a store task; the devices appear only as the columns of a fit verdict.
- **Out by the name:** `serve`, and the `mila tokens` verb of `Vnext.md`'s "Counting a prompt's tokens needs
  Python" — that entry is reread against this scope.

### 4.1 Where the commands live, and their names

**Decided 2026-10-11 (Todd): v0.21.0 renames `Mila/Adaptors/` to `Mila/Apps/`** — the common name, chosen over
`Applications`. Brought forward from v0.22.0's move. The
applications sit at its top level and the shipped utilities under `Tools/` inside it:

| Command | Directory | Ships |
|---|---|---|
| `mila-chat` | `Mila/Apps/Chat` | yes |
| `mila-server` | `Mila/Apps/Server` | yes |
| `mila-store` | `Mila/Apps/Tools/Store` | yes |
| `mila-package` | `Mila/Apps/Tools/Package`, once it ships (section 4.2) | not in v0.21.0 |

Each command has its own directory and its own target. `Mila/Tools/` keeps its rule — none of it ships — so
nothing published lives there, which closes finding 5.

**Decided 2026-10-11 (Todd): lowercase with hyphens everywhere, prefix `mila-`.** The same name on every platform
(`mila-chat.exe` on Windows), so the website, READMEs and cards show one command. The prefix is on what a user
types: commands land on a shared `PATH`, and pip installs console scripts into a shared `Scripts`/`bin`, where
`server` or `store` would collide and say nothing. Directories carry no prefix; `Mila/` is their context. This
reverses `MilaChat.exe` of 2026-09-27 (`Future.md`), which was never built. Developer tools under `Mila/Tools/`
(`ProfileModel`, `ContextProfile`, `Drafting`) keep their names: no user types them.

Today's names against the rule: `mila-chat` and `mila-server` already conform; the CLI target `mila` becomes
`mila-store`; `ExportArtifact` becomes `mila-package` when it ships, and not before — the prefix says "this is your
command", which until then is untrue.

### 4.2 Whether `mila-package` is an end user's tool

**Decided 2026-10-11 (Todd): not in v0.21.0; in future Mila, part of it is, split by the user's act rather than
shipped whole.**

**Today** ExportArtifact's user is Mila publishing its own models. Its modes are a publishing pipeline (export and
quantize, package, validate, compare, fingerprint, fetch, transcode, resume, install, rename). Shipping it would
cost end-user prose for every mode, tests as a published surface, a place in the image or a wheel, and above all a
contract: once users script its flags, they cannot change. No one outside the project gains from it yet — a user
runs what Mila publishes, and quantize-on-load covers "smaller" for FP8 and FP4. It stays developer tooling in
`Mila/Tools/ExportArtifact`, under its current name.

**Future Mila** makes three of its acts a user's, by directions already taken:

1. **Recipes, not packages** (2026-09-27): where a producer publishes an official quantized release, Mila installs
   it by fetching and transcoding locally rather than republishing weights. That runs on the user's machine at
   install, so it belongs in `Mila::Distribution` behind `mila-store install`, not in `mila-package`: the user
   asked to install a model, not to package one.
2. **`Mila::AI` and training:** a component-composed model works like a published one (`Direction.md` 1.1), and
   training returns. A researcher who fine-tunes or composes a model has to save, package, install and run it in
   Chat or MIS. By `CLAUDE.md`'s test — which side of the API someone stands on — that researcher is a user, and
   researchers and engineers are the audience Mila positions itself for.
3. **Publishing stays Mila's own.** Uploading, model cards and comparison against the original are what Mila does
   as a publisher; a researcher publishing their model uses HuggingFace's tools. None of it ships.

| The user's act | Goes to | When |
|---|---|---|
| Install: fetch a producer's release, transcode locally, install from a directory | `Mila::Distribution`, `mila-store install` | with the recipes work |
| Produce their own model: convert, quantize, package, validate | `mila-package`, shipped, `Mila/Apps/Tools/Package` | when `Mila::AI` and training reach users (v0.22.0 or later) |
| Publish: compare, fingerprint, upload, model cards | `Mila/Tools/`, never shipped | stays |

The `mila-package` that ships is therefore smaller than ExportArtifact, and the moment it ships is decided with the
`Mila::AI` and training work, not by this review.

## 5. To be resolved

- **How a user who installed only MIS gets a model. Decided 2026-10-12 (Todd): MIS carries its own model
  verbs**, as Chat does with `/model`; section 3's shape (a) is not taken. Built at `+62` (section 7).
- **Whether the native side needs `mila-store` at all**, once each application carries its own model verbs. The
  image does: its `install` verb exists because a quick start, a Dockerfile or a CI job cannot type at Chat's prompt
  (`Docker/entrypoint-runtime.sh`).
- **Where Chat's fit verdict is computed.** `Chat.Footprint.ixx` and `Chat.FootprintVerdict.ixx` are in Chat. If
  the verdict is computed there rather than by `planDeployment`, `mila-store` would be a second copy; moving it into
  `Mila/Src` is then the work behind the fit column, and needs agreement.
- **One vocabulary across surfaces.** Chat says `/model list` and `/model <name>`; the tool says `models` and would
  say `show`. Matching Chat (`mila-store list`) teaches one set; keeping `models` leaves the website and the image as
  they are.
- **Whether running HuggingFace weights needs a converter.** Not verified. If loading upstream weights needs one of
  the Python converters in `Mila/Tools/Converters`, quantize-on-load — a supported user path since 2026-09-14 — has
  no shipped way in, and converting is a user's act today, putting part of section 4.2's "produce" row in v0.21.0.
- **`show <name>` and the fit column** wait on the two items above them: where the verdict is computed, and which
  words the tool uses.

**Stale, found in this review and deleted 2026-10-11:** `Vnext.md`'s "The `mila` tool has no `pull` verb".
`mila install` is the pull (`Cli.ixx:187`, `ModelResolver::pull`).

## 6. As built, `0.21.0-dev+58`

Admitted to v0.21.0 as an internal fix (`BACKLOG.md`), which also answers section 5's release question for this
part.

- **Moves.** `Mila/Adaptors/Chat` to `Mila/Apps/Chat`; `Mila/Adaptors/Inference/Server` to `Mila/Apps/Server`, with
  the harness guide that was `Inference/README.md` now `Server/docs/harnesses.md`; `Mila/Tools/Cli` to
  `Mila/Apps/Tools/Store`; `Mila/Tests/Adaptors` to `Mila/Tests/Apps`. The gate is `MILA_ENABLE_APPS`, and the
  area tag `apps`.
- **`mila-store`** (`Store.ixx`, module `Mila.Store`) builds beside `mila-chat` at the build root, on any backend:
  the CPU-only presets keep the apps on so it proves it builds without CUDA or HTTP. `install` takes a name or a
  package directory — a directory is copied, not moved, since it is the user's; `models` refuses an unknown
  operand; `remove` reads the record rather than `locate`, which hides the incomplete model a user most needs to
  remove; `usage --clean` discards interrupted downloads too, because the user asked for the space; `--version`.
- **Removed:** `serve`, and with it the launcher, the virtual-environment lookup, `MILA_APP_DIR` in this tool, and
  main.cpp's per-platform image directory. The image's entrypoint keeps its own `serve` verb.
- **Renamed in place:** the website's Get Started and Docker tab, Chat's shell hints and `--help`, the C++ quick
  start's refusal, the runtime and devel images (`/usr/local/bin/mila-store`), the devel image check, the Qwen FP4
  card in the tree (the published card changes with its next upload).

## 7. As built, `0.21.0-dev+62`

- **`mila-server models [--online]` and `mila-server install <name>...`**, in `cli.py`; the bare command still
  serves and now refuses an argument it does not read, rather than dropping it. Loading still never pulls; the
  refusal names `mila-server install <name>`.
- **The fit verdict is the library's.** `models` asks the planner about the card `MILA_DEVICE_INDEX` names, the
  one the server would serve on, through two binding functions over `Mila::Deployment`:
  `stored_weight_quantization` and `fitting_weight_quantizations`. The words are `mila-store`'s.
- **`list_hub_models` returns only repositories with a manifest**, as its documentation said and `mila-store`
  already did; it returned every repository the owner publishes.
- **Not built:** `show` and `remove`.
