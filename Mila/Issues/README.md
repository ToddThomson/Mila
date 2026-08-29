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

`Contributor.md` and `Declined.md` are sanctioned but not created — a category file appears when
it has its first entry. Empty files invite the same "where does this go" paralysis at triage that
`Untriaged.md` exists to remove at capture.

## Untriaged rules

**One line.** `<what> — file:line`. BACKLOG's three-line rule is right for a *commitment*; an
observation has not earned three lines. It gets them on promotion, not before.

**Factual, not speculative-derogatory.** This directory is public, which is correct — it is a
developer surface, and a reference implementation that shows its working should show the parts
not yet worked. "Looks sketchy" is both a bad entry and a bad thing to publish about your own
code. `parse routes any '[' into the tool-call path — Chat.ToolCallParser.ixx:63` is neither.

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
