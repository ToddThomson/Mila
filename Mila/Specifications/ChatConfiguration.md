# Chat Configuration

Layered Resolution of Defaults and Overrides for the Chat Adaptor

---

## 1. Overview

Chat reads a single session JSON file, or nothing. There is no merge: a user who wants one
setting changed writes a whole file and silently inherits whatever the compiled defaults
happen to be. When no file is found, the defaults that take over are not a working
configuration.

Four defects measured on 2026-08-14 all trace to this design rather than to any one value:

- Started outside its own directory in a container, Chat found no config, `context_length`
  defaulted to zero, the load failed, and the process ended in `terminate called without an
  active exception`.
- With defaults in force, Gemma opened at **512** context while effort level 3 budgets **~512
  tokens** of reasoning. Every round ended on `length` with no stop token, on a card reporting
  9.86 of 11.99 GB used — there was room, and nothing asked for it.
- The shipped `session.json` pointed at `assistant.json`, which carries a `get_weather` tool,
  so a 3B answered "why is the sky blue" with a refusal and a function name. The *development*
  config, `session-dev.json`, pointed at the tools-free prompt. The roles were inverted.
- A model installed into a fresh store is not the model Chat opens with, so the two-command
  evaluation path does not reach an answer.

This specification replaces the single-file read with layered resolution, splits family facts
from model facts and moves the model half out of the adaptor entirely, makes the default
context a measurement of the user's card rather than a constant, gives the system prompts a
resolution rule that does not depend on the working directory, and gives Chat a command line —
including a one-shot `-p` and an exit-code contract, so that answering a question is something
a script can do and check.

---

## 2. Root Cause of the Container Failures

`Mila/Adaptors/Chat/Src/main.cpp:112` resolves the default config next to the executable when
it is not present relative to the working directory. Its comment states the intent: this
"makes the default model independent of the process working directory."

```cpp
static std::filesystem::path executable_directory()
{
#ifdef _WIN32
    ... GetModuleFileNameW ...
#endif
    return std::filesystem::current_path();
}
```

**On Linux the fallback is a no-op**, because `executable_directory()` *is* `current_path()`.
The guarantee holds on Windows and silently does not hold anywhere else — which is every
container, and every Linux consumer.

This is the reason `Docker/Dockerfile.runtime` documents its `WORKDIR` as "a correctness
requirement, not tidiness", and the reason `Mila/Tools/Cli` changes directory before running
Chat. Both are workarounds for one missing platform implementation.

**Fix:** implement `executable_directory()` on Linux via `/proc/self/exe`. `main.cpp` is a
plain source file, not a module unit, so the existing `#ifdef` is appropriate there. The
WORKDIR requirement and the CLI's `chdir` both become unnecessary once it lands.

---

## 3. Resolution Order

Each layer overrides the previous **key by key**, never file by file.

| Layer | Source | Purpose |
|---|---|---|
| 1 | Family invariants (compiled) | what the architecture can do |
| 2 | Model manifest (`ModelRecord`) | what this checkpoint recommends |
| 3 | User config | how this person likes Chat |
| 4 | Local config (found) | this checkout, or this image |
| 5 | Remembered choice | the model last chosen from inside a session |
| 6 | Command line: flags, and a file named with `--settings` | this run |

Layer 5 covers one key, `model`, and it was the layer this specification originally failed to
write down — it existed in the code, outranked every file, and appeared in no table. Naming it
is most of the fix. It holds what `/model` or `/install` last loaded, because choosing a model
inside a session is an explicit act that should survive it, and a fresh store has nothing to
remember.

Four rules carry the design:

**Layer 1 alone must be a working configuration.** A run with no files present anywhere must
produce a Chat that can answer. That is what fails today.

**Merge, never replace.** Setting one key inherits every other.

**The device is not a layer.** What fits the card is a *value* a key can take — `"auto"` — not
a silent override that outranks what the user asked for. It resolves after the merge, in §6.
A user who writes `8192` gets 8192 and a warning if it will not fit; a user who writes nothing
gets `"auto"` from layer 1 and gets the card measured.

**Overriding is not setting.** Layer 6 applies to one run and writes nothing back: `--model X`
loads X and leaves the remembered choice alone, so the next run without the flag opens on what
it opened on before. Only an in-session `/model` or `/install` writes layer 5.

**A file named on the command line is part of layer 6, not layer 4.** Layer 4 is the config
that is *found* — this checkout's, or this image's. A file named with `--settings` was chosen
for this run exactly as a flag was, so it carries the same rank and outranks the remembered
choice: pointing at a file that names a model and being handed a different one made the file
mean different things depending on invisible state. It writes nothing back either. That is the
whole rule — explicitness of the invocation decides — and it keeps the order above linear
rather than making layer 5 float.

Environment variables are deliberately **not** a layer. `MILA_CACHE_DIR`, `MILA_APP_DIR` and
`MILA_PORT` remain — they are deployment plumbing (where the store lives, which port to bind),
not chat settings. A container that needs a different model passes `--model`.

---

## 4. Implementation

`nlohmann::json` implements RFC 7386 and is already a dependency:

```cpp
json config = familyInvariants( family );
config.merge_patch( modelRecommendations( record ) );
config.merge_patch( userConfig() );
config.merge_patch( localConfig() );
config.merge_patch( rememberedChoice() );
config.merge_patch( flagOverrides( argc, argv ) );

resolveAutomaticValues( config, record, device );   // section 6
```

Recording which layer last wrote each key, alongside the merge, is what makes §7 free.
`resolveAutomaticValues` records itself the same way, so an origin column can say `auto`
rather than leaving a derived number looking like something a file asked for.

**Validation happens once, after the merge and resolution, before anything is constructed.** A
`context_length` of zero is rejected there with the key named. Nothing reachable from a
configuration file may reach `std::terminate`.

---

## 5. Family Invariants versus Model Recommendations

The present code files a deployment judgement under architecture. `defaultContextFor` in
`Chat.ModelCatalog.ixx:229` returns 512 for Gemma, and its own comment admits the reason:

> "A deployment decision rather than a model property: Gemma 4 12B is conservative because its
> KV cache is the primary VRAM lever on a 12 GB card, not because the architecture cannot go
> further."

A published Gemma 4 4B would inherit 512 absurdly; a 27B would inherit it dangerously. The
value is a fact about one checkpoint against one card, wearing a family label. It does not move
down to the model layer either — §6 deletes it, because the card it was a fact about is not
necessarily the card the model is running on.

**Family — compiled into Chat.** True of every checkpoint of that architecture, and not the
publisher's to change:

- `thinking_capable` — the architecture has a reasoning channel
- `streaming_capable`
- `max_context` — the hard ceiling, such as GPT-2's 1024 learned position rows, where a larger
  value indexes past the table and the load fails
- template and tokenizer traits

These are properties of the code that implements the architecture, so they belong beside it.
They are also today scattered across four switch statements in one file — `:229`, `:250`,
`:469`, `:474`. That scattering is not cosmetic: "Gemma is `thinking_capable`" and "Gemma
defaults to 512 context" sit 245 lines apart, and nothing reconciles them. One table per
family puts the contradiction where it cannot be missed.

**Model — carried in the manifest.** `ModelRecord` already carries checkpoint facts
(`instruct`, `weight_quantization`, `architecture`, `base_model`, `minimum_mila_version`). It
gains recommendation:

```
default_context_length          what to open at when the user asks for nothing
maximum_context_length          what this checkpoint was trained to, which bounds "auto"
temperature / top_p / top_k     (optional, when the model card states them)
```

The two context fields are different questions and only one of them is a ceiling. A checkpoint
trained to 8192 does not become a 128K model because the family's rotary embedding admits one,
and `"auto"` must not walk past the training length just because the card has room — so §6
clamps to `maximum_context_length`, while `default_context_length` is only what a user gets
before they express a preference.

**Publishing a model must not require editing a switch statement in a chat adaptor.** The
model knows its own size; it carries its own recommendation. Absent fields fall back to the
family layer, so existing published models keep working and can be republished later.

MIS and the Python binding read the same record, so the recommendation is shared rather than
duplicated per adaptor.

---

## 6. `context_length: "auto"`

Neither a family nor a model default knows the user's card. Gemma 4 12B FP4 at 8192 measured
**11.07 of 11.99 GB** on an RTX 4070 — at the edge. The same model on 24 GB could take far
more; on 8 GB it fits nothing. No constant compiled into an adaptor can be right for all
three, which is why the 512 in §5 exists at all: it is a constant chosen to be wrong safely.

**`context_length` therefore takes a positive integer or the string `"auto"`, and layer 1
supplies `"auto"`.** That is what makes rule one of §3 achievable — a run with no files present
anywhere opens at the largest context the card in the machine can hold, rather than at a number
guessed years earlier on different hardware.

### What auto resolves to

The largest context that fits **comfortably**, bounded above by
`min( family max_context, model maximum_context_length )`.

Comfortably needs a number, because 11.07 of 11.99 GB is a fit by arithmetic and a bad
experience in practice — that is a 92% claim on a card that also drives a display. **Auto
targets leaving the greater of 10% of total device memory or 512 MB free**, measured against
`practicalDeviceBytes`, which already carries the residual the predictor does not model.
A user who writes an explicit number is not held to this: the margin is what auto chooses for
you, not a policy imposed on what you chose.

`predictFootprint` reads the artifact header and allocates nothing on the device. Measured
2026-08-15: the `/models` listing's ten predictions cost about 25 ms in total, a few milliseconds
each, so probing is cheap enough to run at every startup — which matters, because the answer
legitimately changes between runs as other processes take and release memory.

**Search must not be by bisection.** Bisection assumes the footprint rises with context, and
Gemma's does not. Measured on `gemma-4-12b-it-fp4`, RTX 4070, 11.99 GB:

| context | 4096 | 6144 | 8192 | 10240 | 12288 | 16384 | 20480 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|---|---|---|---|
| GB | 9.35 | 9.54 | 9.73 | 9.92 | **9.37** | 9.52 | 9.62 | 9.93 | 10.35 | 11.68 |

The curve rises at 0.095 GB per 1024 tokens to a local peak at 10240, **drops 0.55 GB at 12288**,
then resumes at roughly a quarter of that slope. The shape is consistent with prefill chunking
capping the activation and scratch buffers once the sequence passes a threshold, after which only
the KV cache grows — and Gemma's is mostly sliding-window, so only the eight global layers of 48
grow with context. A bisection against a budget between 9.4 and 9.9 GB lands wherever it happens to
probe: the true answer is past the drop, and half the search space says it is not.

Scan the 1024-token grid instead and take the largest fitting point. At a few milliseconds a probe,
128 points cost well under a second, and the scan is correct on any curve rather than on a monotonic
one.

### What auto resolves to on a 12 GB card

Leaving the greater of 10% or 512 MB free gives a 10.79 GB budget on the card above. Against the
measured curve, `gemma-4-12b-it-fp4` fits between 65536 and 131072 — where the compiled default is
**512**, which is the truncation defect this specification opens with. `Llama-3.1-8B-Instruct-fp4`,
which has no sliding window and grows at 0.22 GB per 1024, lands between 4096 and 8192.

### Fitting in memory is not the same as running well

Measured 2026-08-15, with auto implemented: `gemma-4-12b-it-fp4` resolves to **95232**, loads, and
answers. It is still the wrong number, and the reason is not memory.

`GemmaTransformer::resolvePrefillChunkSize` (`Gemma.ixx:1023`) picks a prefill chunk from the rungs
1024 / 512 / 256 / 128 / 64 rows, against an activation budget of 1536 MB **minus the global KV
term, which grows with context**. So the effective budget shrinks as context grows, and the chunk
walks down the rungs. With the measured row cost of 662464 bytes:

| prefill chunk | needs global KV under | which holds to about |
|---|---|---|
| 1024 rows | 858 MB | 54K context |
| 512 rows | 1197 MB | 76K |
| 64 rows | 1494 MB | 95K |

Past 95232 not even the 64-row floor fits and the transformer warns that it "cannot prefill
efficiently". **Auto picked 95232 — the last context before that warning, where prefill runs at the
64-row floor, sixteen times smaller than the 1024-row chunk.** The independently recorded live
usable ceiling of 49-56K is not a disagreement with the predictor after all: it is the chunk-1024
boundary in the table above, measured from the other direction.

The memory budget is therefore necessary and not sufficient. Auto should take the largest context
that still prefills at a full chunk, which on this card is roughly 54K rather than 95K.

**Chat cannot compute that.** `resolvePrefillChunkSize` is private to the transformer, and the
adaptor has no way to ask what chunk a context would get. The blocker is a runtime capability — a
model answering "what prefill chunk would this context use", beside `getRequiredMemory`, which
already answers the memory half of the same question. Until it exists, auto's number is a memory
answer to a question that also has a throughput half.

### When auto cannot answer

Derivation fails honestly rather than silently: no CUDA device, a family with no footprint entry
point, an unreadable artifact. Auto then falls back to `default_context_length`, and to the
family floor if the manifest carries none.

**Auto reports what it resolved to, and why, on the startup line that already prints** — not
behind a diagnostic flag. A derived number with no provenance is the same complaint this
specification opens with about 512; the answer is to put the provenance where every user
already looks rather than where a user must know to ask:

```
Context 8192 (auto, 12.0 GB device)
Context 4096 (auto -> model default, no CUDA device)
```

The same pair of facts appears in the `--output-format json` payload (§9), which covers the
scripted case. Nothing else is needed: those are the two places a resolved context is read.

### The predictor must be able to say why it has no answer

`/models` reported `--` for `gemma-4-12b-it-fp4` on 2026-08-14 — "a load that would work but goes
unmeasured", per the legend at `Chat.ModelCatalog.ixx:727`. **That symptom no longer reproduces:**
measured 2026-08-15 at 512, 4096 and 8192 context, the row reports 8.32, 9.35 and 9.73 GB. No commit
in between touched `Chat.Footprint.ixx` or `GemmaModel.ixx`, so what changed was the input rather
than the code — and `GemmaModel::getRequiredMemory` throws on a zero context length
(`GemmaModel.ixx:201`, and `LlamaModel.ixx:196` alongside it), which the bare `catch` turned into a
dash. That is the same zero-context state §2 blames for the container failure, and phase 4's layer 1
removed it.

Two things the episode leaves behind, and they are what this phase does:

The reason was **discarded**, so a symptom whose cause was already documented in §2 was diagnosed as
a second, separate defect. `predictFootprint` now returns the reason alongside the footprint. The
contract at `Chat.Footprint.ixx:174` — "a pre-flight must never be the thing that stops a model from
being tried, so a failure here is silence rather than a throw" — is correct for the pre-flight and
wrong for auto, which must know why it got nothing in order to report the chain above. The silence
stays for the caller that wants it.

The Gemma branch had **no precision guard** where Llama's has one, so an FP32 deployment was answered
by the BF16 instantiation: a wrong number rather than a declined question. Nothing exercises that
path today, since every installed Gemma is BF16-activation, which is why it went unseen.

The contract at `Chat.Footprint.ixx:174` — "a pre-flight must never be the thing that stops a
model from being tried, so a failure here is silence rather than a throw" — is correct for the
pre-flight and wrong for auto, which must know why it got nothing in order to report the chain
above. The entry point gains a failure reason; the existing silent behaviour stays for the
caller that wants it.

---

## 7. File Layout

**The session files shrink; they do not collapse.** `session.json`, `session-dev.json`,
`session-llama-3b-fp4.json` and `session-llama-8b-fp4.json` are near-identical today because
each one must be complete — there is no merge, so every file repeats every key. Merge removes
that duplication at the source: once layers 1 and 2 supply everything, a per-model file is the
two or three keys that actually differ, and `--settings` (§9) selects one.

No profile mechanism is introduced. Named sections plus a `current` key plus a second selector
would be a second way to choose a configuration, solving a duplication problem that §3 has
already solved.

**Config root, beside the store root.** `resolveStoreRoot()` in
`Mila/Src/Distribution/ModelStore.ixx:87` already resolves the cache per platform. A
`resolveConfigRoot()` mirrors its shape with configuration directories:

| Platform | Path |
|---|---|
| Windows | `%APPDATA%\Mila\chat.json` |
| Linux | `$XDG_CONFIG_HOME/mila/chat.json`, else `~/.config/mila/chat.json` |
| macOS | `~/.config/mila/chat.json` |

`%APPDATA%` and not `%LOCALAPPDATA%`: settings are small and personal and should roam, while
the store correctly stays in Local because model blobs are large and machine-specific.
Different variable, deliberately. macOS takes the Linux path for symmetry — Apple's own
convention is `~/Library/Application Support`, worth revisiting only if a Metal backend ever
makes macOS a real target.

---

## 8. Prompt Files, and the End of `Data/`

`Mila/Adaptors/Chat/Data/` is a relic of Chat's first weeks and holds two unrelated kinds of
file: four `session*.json` configurations, which §3 through §7 replace outright, and four
`assistant*.json` system prompts, which are **content** and survive. Once the configurations
leave, `Data/` is a directory named after nothing.

### Prompts are content, but the reference to them is a setting

`system_prompt_path` is an ordinary merged key — a local config may set it and a flag may
override it. What is not ordinary is resolving the value, because the merge draws from files in
three different directories and a relative path means something different in each.

**A relative path resolves against the directory of the file that set the key.** Not the
working directory, and not the executable. This is the rule tsconfig, Cargo and every
include-path system settles on, and it is the only rule that survives the config root moving to
`%APPDATA%`: a user writing `prompts/mine.json` in their own `chat.json` means their own
directory, while a shipped local config naming a prompt beside itself means the install tree.
Recording the origin of each key (§4) is what makes this implementable — the resolver needs to
know which file a value came from, which the merge already knows.

Two cases sit outside that rule, both deliberately:

- **A bare name with no separator and no extension is a named prompt**, resolved through a
  search path: the config root's `prompts/` first, then the install's `Prompts/`. So
  `"system_prompt": "assistant"` is written once and works from a checkout, an install and a
  container alike, and a user shadows a shipped prompt by putting their own next to their
  config. This is what shipped configs should use.
- **`--system-prompt` on the command line resolves against the working directory**, because a
  user typed it there and tab-completed against it.

A named prompt that cannot be found is rejected by name at validation (§4), alongside
`context_length`. It is not silently replaced by a compiled-in default: a user who asked for a
persona and got a different one has no way to discover that from the transcript.

### The files, renamed for their roles

`main.cpp:293` today resolves prompts relative to the executable "where `Data/` is copied", and
the current names invert what they are. `assistant.json` is not the assistant — it is a
weather-tool demonstration, and it is what made a 3B refuse to explain the sky. Meanwhile
`assistant-chat.local.json` is the shipped default, wearing `.local`, which everywhere else in
this codebase and most others marks a file a user wrote and git ignores.

| Today | Becomes | Role |
|---|---|---|
| `Data/assistant-chat.local.json` | `Prompts/assistant.json` | the shipped default: a plain assistant, no tools |
| `Data/assistant.json` | `Prompts/tools-weather.json` | the tool-calling demonstration |
| `Data/assistant-tools.json` | `Prompts/tools.json` | full tool set |
| `Data/assistant-tools-one.json` | `Prompts/tools-single.json` | one-tool variant |
| `Data/session*.json` | (deleted) | shrink to their differing keys, in the config root, §7 |

The default prompt is then the one whose name says default, and the name that reads like a user
override belongs to a user. `Data/` is removed rather than emptied.

This also finishes what §2 starts. After phase 1, Chat no longer needs the working directory to
find its configuration; after this section it no longer needs it to find its prompts either, and
the last reason `Mila/Tools/Cli` changes directory before launching Chat is gone.

---

## 9. Command-Line Flags

Chat accepted two flags before this work, `--config` and `--help`; every other setting was
reachable only by writing a file. This is not an extension of a flag set, it is the creation of
one, so the surface is decided whole rather than grown.

The harness is invoked directly rather than through the `mila` CLI, which carries no `chat`
verb: `mila-chat` is a binary beside `mila` and needs no front door, while the server is a
console script inside a virtual environment and does. That is also why the binary is named for
what a user types rather than for its build target.

| Flag | Value | Notes |
|---|---|---|
| `--model` | store name or catalog alias | the name `install` accepted must work here |
| `-p` | prompt text | one shot: answer, print, exit. Never interactive |
| `--settings` | path to a JSON file | replaces `--config`; layer 6 of §3, not layer 4 |
| `--context-length` | integer or `auto` | spelled as the key it overrides |
| `--system-prompt` | name or path | resolved against the working directory, per §8 |
| `--output-format` | `text` (default) or `json` | meaningful only with `-p` |
| `--version` | — | |
| `--help`, `-h` | — | |

`--model` must accept the **store name** (`Llama-3.2-3B-Instruct-fp4`), not only the catalog
alias (`llama-3b`). The name a user has just typed to `install` is the store name, and the
two-command evaluation path fails at the second command if the flag will not take it.

`--context-length` accepts `auto` because layer 6 must be able to express everything layer 1
can. A flag set that cannot say what the compiled default says is an incomplete override layer.

### Flags produce a patch, not assignments

`flagOverrides( argc, argv )` in §4 returns a JSON object — `{"context_length": 8192}` — which
is merged like any other layer. It does not assign to `ChatConfig` members. Three things follow:
adding a flag is one row in a table, origin recording is free rather than a parallel mechanism,
and no bespoke flag-to-config path is built for phase 3 to delete. Eight flags do not justify a
dependency; a table of specifications and a hand-written loop is the whole parser.

### Output discipline

**Standard output carries the answer and nothing else.** The banner, the spinner, warnings and
the auto-resolution line of §6 all go to standard error, so that `mila chat -p "..." > answer.txt`
yields an answer rather than an answer wearing a welcome box. With that split, plus suppressing
the spinner when standard output is not a terminal, no `--plain` flag is needed — there is
nothing left for it to suppress.

`--help` and `--version` must answer in a broken environment. `printUsage` currently reads the
model store (`main.cpp:65`), and a container user's first command is often `--help`, before any
model exists.

### Exit codes

`-p` is only useful to a script or a CI job if failure is legible without parsing prose.

| Code | Meaning |
|---|---|
| 0 | answered, or `--help` / `--version` |
| 1 | anything else |
| 2 | usage error: unknown flag, missing value |
| 3 | configuration invalid — names the offending key, per phase 2 |
| 4 | model not found in the store |
| 5 | model load failed: device memory, artifact, version floor |

**A generation that ends on `length` is not a failure and does not get a code.** It is a
successful run that was cut short; giving it one would make every caller special-case a success.
It is reported through `finish_reason` under `--output-format json`, and on standard error
otherwise.

This contract cannot be honoured through `Mila/Tools/Cli` as it stands. `runProgram`
(`Cli.ixx:72`) builds a command string and hands it to `std::system`, so the code observed is
the shell's rather than Chat's — and the same construction concatenates arguments into that
string with no escaping (`Cli.ixx:86`). That is survivable for `install gpt2-small` and is not
survivable for `-p`, which is exactly the flag that turns arbitrary user prose into a shell
argument: on Windows `std::system` routes through `cmd.exe`, which expands `%...%`, and a prompt
containing a quote or an ampersand either mangles or executes. Launching with an argument vector
— `CreateProcessW`, `posix_spawn` — fixes the quoting and the exit code together, and is a
prerequisite for both.

---

## 10. Phasing

The flag set of §9 does not land as one item. Half of it fixes live defects on today's
single-file read and half depends on the merge, so it is split across the phases that support
it rather than held until they all exist.

| Phase | Work | Why here |
|---|---|---|
| 1 | `executable_directory()` on Linux | removes the cause of four symptoms, and retires the WORKDIR requirement and the CLI's `chdir` |
| 2 | Validate before construction; reject `context_length <= 0` by name; no path from a config file reaches `std::terminate` | makes a bad value legible instead of an abort, and is what the exit codes of phase 3 report |
| 3 | Argument-vector launch in `Mila/Tools/Cli`; `--model`, `-p`, `--output-format`, `--version`, `--settings`; the exit-code contract | fixes live defects without waiting for the merge; the launch fix is a prerequisite for `-p` and for any exit code at all |
| 4 | Family invariants table; layered merge with `merge_patch`, recording origins; `--system-prompt` and `--context-length` join as patch producers | layer 1 becomes a working configuration; origins are what §6 and §8 both resolve against, and what the remaining flags plug into |
| 5 | Give `predictFootprint` a failure reason; close the Gemma precision guard | auto has to report what it fell back to and why, which a predictor that discards its reason cannot support |
| 6 | `context_length: "auto"` (§6), as the layer 1 default and a `--context-length` value | the constant that caused the truncation defect stops existing |
| 7 | `default_context_length` / `maximum_context_length` in `ModelRecord` | new models stop requiring adaptor changes, and auto gains its upper bound |
| 8 | Prompt resolution and the `Prompts/` rename (§8) | needs phase 4's origins; retires `Data/` |
| 9 | `resolveConfigRoot()` | tidying, and the last thing tying configuration to the install tree |

Phases 1 through 3 are independent of the rest and fix defects that exist today. Phase 6 is the
one that closes the Gemma truncation defect: auto on a 12 GB card resolves far above 512, so a
reasoning budget of ~512 tokens stops being the entire context. Phase 7 touches Model
Distribution, which is a deliberate v0.20 carve-in.

Until phase 7 lands, auto's upper bound is the family ceiling alone, which overshoots a
checkpoint trained shorter. That is acceptable in the interim — the models in the store are
long-context — but it is the reason 7 does not drift.

**One dependency lies outside this specification.** `mila install <name> && mila chat -p "..."`
reaches an answer only if a store holding exactly one model loads it. That defect is filed
separately; phase 3 delivers `-p` either way, but the one-command evaluation path needs
`--model` on every invocation until it is settled.
