# Mila — Backlog

**Work committed to the release in flight, and nothing else.** Narrative and success criteria are in
[ROADMAP.md](ROADMAP.md); everything upstream of the commitment is in
[`Mila/Issues/`](Mila/Issues/README.md). Completed work is in the git history.

**Admission:** name the ROADMAP success criterion that fails if this never ships. If you cannot,
it belongs in `Mila/Issues/`.

Each `###` bucket is a v0.20 theme, its name matching the ROADMAP section — the only join.

**Entry shape**, one level deeper than the same shape in `Mila/Issues/`:

```markdown
#### Llama throws away its long-context scaling factor

`open` · `llama`

The load path reads `rope_scaling` from the model metadata and discards it — the
`.withRoPEScalingFactor()` call at `Llama.ixx:703` is commented out, for a reason recorded as
unclear. 3.1 8B cannot reach the context length it advertises.
```

The **heading states the problem** and has to read cold: no term that exists only inside Mila.
The **metadata line** carries status — `open`, `in progress`, `done` — then area tags from
[Tags.md](Mila/Issues/Tags.md); status never appears in the prose. The **body** carries whatever
detail the work needs and ends in an anchor, unless the finding is an absence, in which case say so
rather than inventing a location.

**The gate is the entry count, and it only goes down.** A release in flight burns down, so an
addition is paired with a removal or it is a deliberate admission that scope grew. Past roughly
forty entries this is a wishlist, not a release.

**Done means deleted**, in the same commit as the work — `done` is a working-tree marker and is
never committed.

---

## Current release (v0.20.0)

### Packaging & Distribution

#### The dev container has not been shown to build the bind-mounted tree

`in progress` · `build`

`getting-started.md` §4 tells a reader to configure and build under `/mila`, the repository bind
mount. Validated on a clang-21 + gcc-15 host at CUDA 13.3 into a container-local directory only. The one
known failure there was CUTLASS's clone step, and CUTLASS is no longer fetched. Runs during the
v0.20.0 release, alongside the consumer builds.

---

### Consumer & Contributor Surface

#### There is no guided reading path through the source

`open` · `docs`

Mila's positioning is the stack you can read, and nothing shows a reader where to start. One token's
journey — embed, attend, sample, decode — through the real source, followable by a strong C++
developer unaided. No anchor: the finding is an absence.

---

### Model Distribution

#### The published model cards tell users to run `/install` and `/models`, which Chat does not have

`open` · `distribution` · `docs`

Chat's commands are `/model install <name>`, `/model load <name>` and `/model list`. The card
sources in the repository are correct; the live copies on huggingface.co only change when a model is
re-published, and they are what a new user reads *before* they have Mila at all. Fold the card
refresh into the next publish of each: `Mila/Tools/ExportArtifact/ModelCards/`.

---

### Product Family — Adaptor Validation

#### Codex CLI has not been driven against Gemma's reconciled tool grammar

`open` · `gemma` · `adaptors`

The Codex CLI round-trips the release criterion names were validated before the native grammar was
reconciled to Google's canonical template. Re-run plain-chat, single-tool and tool-result-resume
through MIS against the current grammar.
