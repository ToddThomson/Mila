# Tags

The closed set. **A tag not on this page is not a tag** — an entry carrying one is malformed, and
the fix is to pick from here or to amend here first.

Tags describe **area and constraint, never disposition.** Disposition is already the filename:
`Future.md`, `Contributor.md`, `Declined.md`. Encoding it twice is how the two drift apart and how
the categories stopped meaning anything the last time.

Where they apply: `Future.md`, `Contributor.md`, `Declined.md`. **Not `Untriaged.md`** — one line
per entry leaves no room, and an observation has not earned a taxonomy. **Not `BACKLOG.md`**, which
has its own set; see the bottom of this page.

---

## Area — what part of the system

Fixed by the shape of the tree and the ROADMAP themes, not earned by usage. A rarely-used area tag
is a statement about where the work is, not a tag that needs retiring.

| Tag | Means |
|---|---|
| `models` | A model family — its chassis, config, or protocol. |
| `quantization` | Weight and KV formats, packing, the offline fitting pipeline. |
| `training` | The backward path, optimizers, loss, data loaders. |
| `tokenizer` | BPE, pre-tokenization, vocabularies. |
| `api` | The public `import Mila;` surface — names, signatures, coherence. |
| `architecture` | Cross-cutting structure: dispatch, the type axes, device backends. |
| `build` | CMake, presets, compilers, warnings, platform support. |
| `ci` | Workflows, gates, runners, release mechanics. |
| `docs` | Doxygen, README, specs, the website, model cards. |
| `adaptors` | Chat and MIS. |
| `binding` | The Python projection. |
| `perf` | Throughput, bandwidth, memory residency. |
| `observability` | View and Observe. |
| `distribution` | The store, manifests, fetching, publishing. |

## Constraint — what makes it costly or gated

These are earned: a constraint tag must be true of at least three entries, or it is not carrying
its weight and should be said in prose instead.

| Tag | Means |
|---|---|
| `mila-src` | Touches the frozen library. **Waits by default** — this is the single most common reason an entry is in `Future.md` rather than `BACKLOG.md`. |
| `breaking` | Changes a published wire format or a public API. Implies a republish, a migration, or both. |
| `blocked` | Has a named blocker, stated in the body. Never a vague one. |
| `measured` | Carries a number that cost GPU time. **Do not delete this entry without rehoming the number** to its owning spec. |

`measured` is seeded deliberately below the three-entry threshold — it is on one entry today. It
earns the exception because it is the only tag that guards against a specific, already-observed
failure: deleting an entry and losing a measurement with it. Reconsider it at the next triage; if
`Declined.md` has not grown into it by then, it goes.

## Sequencing — when

| Tag | Means |
|---|---|
| `gate` | Blocks something named in the body. Not "important". |

**`next` was retired when `Vnext.md` was created.** It meant "first work after the current release
tag" — a disposition, and disposition is the filename rather than a tag. It was on one entry, below
the threshold anyway. Where sequencing *within* `Vnext.md` matters, say it in the body, which is
more precise than a tag and cannot silently contradict the file it sits in.

---

## Rules

**Order:** area, then constraint, then sequencing. Separator is ` · `.

**Cap of four.** An entry needing five tags is doing too much and should be two entries — the tag
count is a usable proxy for scope creep in a way the line count is not.

**Adding a tag** amends this page in the same commit. Area tags follow the tree; constraint tags
must be true of three entries first.

**Retiring a tag:** at triage, a *constraint* tag on fewer than two entries goes, and its meaning
moves into the prose of the entries that had it.

---

## `BACKLOG.md` has a different set, deliberately

BACKLOG's tags — `[gate]`, `[crash]`, `[net-new]`, `[decoupled]` — describe **properties of
committed release work**: whether it blocks the tag, whether it reproduces as a crash, whether it
is authored from scratch, whether it is off the critical path. Those questions only exist once
something is committed to a release, which is why they do not appear here.

The two sets share one word. `[gate]` in BACKLOG means *blocks the release*; `gate` here means
*blocks the thing named in the body*, because nothing in `Future.md` blocks the release in flight
by definition. That is a real difference and not worth collapsing.

One registry, two sets, stated in one place — so the next person does not find a third.
