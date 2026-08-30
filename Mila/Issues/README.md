# Mila — Issues

Everything noticed but not yet committed to a release. `BACKLOG.md` at the root holds work
committed to the release in flight; this directory holds everything upstream of that decision.

The split exists because **capture and triage have different budgets.** Finding something
mid-task leaves seconds for judgement, and a categorised list demands a decision that cannot be
made in that window — so everything defaults into the current release, which is a commitment
nobody meant to make. Writing to `Untriaged.md` needs no decision. Triage supplies it later.

## The flow

```
  found while working  ─┐
                        ├─→  Untriaged.md  ──triage──→  BACKLOG.md  (committed to this release)
  reported on GitHub   ─┘                      └─→  Future.md   (real work, later cycle)
                                               └─→  a category below
                                               └─→  deleted
```

Triage runs at each `beta.N` / `rc.N` increment. It is an event, not a mood: open `Untriaged.md`,
give every line a destination, and leave the file shorter than you found it.

**Promotion is one-way.** An item that reached `BACKLOG.md` never returns to `Untriaged.md` — that
file is lossy, and demoting a considered item into it puts it on a delete-at-tag timer. If
committed work turns out not to belong in the release, it goes to `Future.md` or a category below.

## Categories

A category names **what happens to an item**, never what it is about. Subject is already
recoverable by grep; disposition is not. Categorising by subject would rebuild BACKLOG's theme
buckets one level up and reproduce its failure mode.

- **`Untriaged.md`** — captured, not yet judged. Lossy; see below.
- **`Future.md`** — real work, wrong cycle. Flat and coarse by design: detailed tasking happens
  only when an item promotes into a release.
- **`Contributor.md`** — good-first-issue shaped. The outbound queue: these are what get mirrored
  *to* GitHub Issues with a label when someone asks how to help.
- **`Declined.md`** — considered, not doing, with the reason. Cheaper than rediscovering the
  argument.

## The entry format

**One shape in every file here.** Heading, metadata line, body:

```markdown
## <Title — the thing itself, as a noun phrase>

<metadata line>

<Body.>
```

Only the metadata line differs by file, because only its question differs:

| File | Metadata line | Body is |
|---|---|---|
| `Untriaged.md` | `<anchor> @ <sha>` — where to find it | what you were doing when you noticed |
| `Future.md` | tags | what it is and why |
| `Contributor.md` | tags | what it is and why |
| `Declined.md` | tags | the reason, and the measurement behind it |

The heading is what makes a file navigable — an editor's outline pane and the markdown TOC both key
on it, and entries collapse.

Every file opens the same way too: `# Name`, a paragraph saying what it holds, a paragraph carrying
the one rule that matters plus the pointers here and to `Tags.md`, then `---`, then entries. **No
file carries its own rules inline** — they live on this page, so there is one place to change them.

Tags come from the closed set in [Tags.md](Tags.md) and nowhere else. That page also records why
`BACKLOG.md` keeps its own four tags rather than sharing these.

## Untriaged rules

**Facts may grow; judgement may not.** Whether something blocks the release is triage's question,
and answering it at capture is what put everything into `BACKLOG.md` in the first place. The body
is the trigger — what you were doing — not the argument.

**The anchor** is `file:line`, or a test name, a repro command, a model and prompt — whatever makes
it re-findable. Some findings have no line to cite: an absence, a pattern spread across files, a
runtime behaviour. An entry distorted into a fake location is worse than one anchored honestly.

**Append `@ <sha>`.** A line number rots on the next edit; `git show <sha>:<path>` still returns the
exact text afterwards. It costs nine characters and no judgement — write the current HEAD. Without
it, an entry that survives one refactor points at nothing and gets deleted at triage for being
undecodable, which is the lossiness rule destroying signal rather than noise.

**Name the symbol, not just the location.** `matchesPath`'s glob outlives
`CompositeComponent.ixx:405`, and both together cost one clause.

**Lossy by design.** An entry still in `Untriaged.md` at the release tag is **deleted, unexamined** —
not re-triaged. If nobody promoted it across a whole cycle it was noise, and rediscovering it
later costs less than carrying it. This rule is what stops this file becoming a second backlog,
and it is the one that will feel wrong.

**Lossiness applies to `Untriaged.md` alone.** Every other file here keeps its own discipline, or the
pile simply moves.

## GitHub Issues

GitHub is the front desk; this directory is the work queue. Someone with no repo access files
there and gets a notification when it is fixed, which a file in a repository can never do.

Two rules keep the two from drifting apart:

**The funnel is manual.** A human decides which reports earn an entry. Public projects
attract "doesn't build on my machine", and an automatic pipe would fill `Untriaged.md` with
exactly the noise it exists to keep out of BACKLOG.

**For anything user-reported, the GitHub issue stays the record and the entry is only a
pointer to it** — one line, an issue number, no substance copied. That keeps GitHub authoritative
for items with a person waiting, and it makes the lossiness safe: deleting a pointer destroys
nothing, because the issue is still open with its thread and its reporter intact. A self-found
note may evaporate; a user's report may not.
