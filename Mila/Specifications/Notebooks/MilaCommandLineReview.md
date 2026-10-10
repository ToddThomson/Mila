# Review of the `mila` Command Line Utility

**Status:** Review, 2026-10-10, at `0.21.0-dev+57`. Findings, one recommendation, and one decision (section 4: a
published utility is for the end user only). **To be
resolved first: what the utility encompasses** (section 4). Until that is settled no finding here is admitted to
a release, and the one recommendation (section 3) depends on it.

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
   `Mila/Applications/`; where the utility moves is not recorded.
6. **A naming decision is on record that this review's question contradicts.** `Mila/Issues/Future.md`, "`ExportArtifact`
   needs a name, subcommands, and its store verbs handed back to `mila`", records names decided 2026-09-27: this
   tool `mila-cli`, ExportArtifact `mila-package`, Chat `MilaChat.exe`. Todd's question of 2026-10-10 uses `mila`
   and `mila serve`, as the executable is named today.
7. **Two tools install into the store.** `ExportArtifact --install` installs a local package; `mila install` pulls a
   published one. Chat (`Chat.ModelCatalog.ixx:387`) and the server (`model_worker.py:90`) tell a user without a
   model to use one or the other (the same `Future.md` entry).

## 3. How a pip user would get `mila`

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

## 4. To be resolved: what the utility encompasses

Each answer changes the findings above, which is why it comes first.

**Decided 2026-10-10 (Todd): a published utility is for the end user only.** Whatever ships carries the verbs a
person running Mila needs, and nothing a developer of Mila or a model producer uses. That answers two of the
questions below and bounds the rest.

- **The store only, or the store and the applications' front door?** Today it is both: two store verbs and a
  launcher. Whether `serve` belongs in it is the question that started this review. `remove` and `usage`
  (finding 4) are end-user verbs.
- **Packaging — ExportArtifact's verbs. Out, by the decision above:** packaging, validating and transcoding are a
  model producer's work, which is the `mila-package` tool of finding 6. Finding 7 remains: an end user who installs
  a package someone handed them, rather than a published one, has no verb for it in this utility.
- **Publishing. Out, by the decision above:** `Tools/Publishing/publish_model.py` uploads a package, which is a
  producer's act; the library itself never uploads.
- **Chat.** Settled out on 2026-08-15; listed so the answer is reread against the scope chosen here, not reopened.
- **The name** — `mila`, or the `mila-cli` of 2026-09-27.
- **Where it lives** after v0.22.0's move to `Mila/Applications/`.
- **Which release** — whether any of it is v0.21.0 work, and against which criterion; the nearest is Applications
  Ship's.
