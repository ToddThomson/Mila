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
  reported on GitHub   ─┘                      └─→  Vnext.md    (seeds the NEXT backlog)
                                               └─→  Future.md   (real, zero commitment)
                                               └─→  a category below
                                               └─→  deleted

  and at each publish:  one or two THEMES lifted out of Vnext.md  ──→  the next BACKLOG.md
```

**Triage runs at the release-prep commit, and whenever `Untriaged.md` passes twenty entries.** It is
an event, not a mood: open `Untriaged.md`, give every line a destination, and leave the file shorter
than you found it. The trigger used to be each `beta.N` / `rc.N` increment, which a one-tag cycle
deleted rather than renamed — so half of it is now the one release event that still exists, and half
is a size bound, because a cycle can run long enough for the file to grow past what a single pass
can hold in view. **A full pass beats batches of five**: two entries are only visibly the same entry
with the whole file in front of you.

**Promotion is one-way.** An item that reached `BACKLOG.md` never returns to `Untriaged.md` — that
file is lossy, and demoting a considered item into it puts it on a ninety-day timer. If committed
work turns out not to belong in the release, it goes to `Future.md` or a category below.

## Categories

A category names **what happens to an item**, never what it is about. Subject is already
recoverable by grep; disposition is not. Categorising by subject would rebuild BACKLOG's theme
buckets one level up and reproduce its failure mode.

- **`Untriaged.md`** — captured, not yet judged. Lossy; see below.
- **`Vnext.md`** — the **standing pool a release is drawn from**, and the one file here with no
  timer. It drains by extraction rather than expiry: at each publish one or two *themes* are lifted
  out of it and become that release, and what is not lifted stays. An item here carries a real
  intention to do it in some near cycle — a weaker claim than `BACKLOG.md`'s and a stronger one than
  `Future.md`'s. A shortlist, not a plan: tasking happens on promotion, when there is a ROADMAP
  section whose criteria an item can actually face.
- **`Future.md`** — real work carrying **zero commitment**, some of which will never be scheduled.
  Flat and coarse by design.

The line between those two is the only judgement the split asks for. "We mean to do this next
cycle" is `Vnext.md`; "someday, if the hardware or the reason arrives" is `Future.md`.
- **`Declined.md`** — considered, not doing, with the reason. Cheaper than rediscovering the
  argument.

**There is no file for work to hand to a contributor, and there is not going to be one.** An item
worth offering is filed on GitHub with the `good first issue` label, by the triage pass that decided
it — one step, not a local queue that drains into one. The queue that existed held 21 entries and
released 2 in two months, and what stayed behind was dead code, stale anchors and doc rot, because
writing to a file has no audience and the bar drifts with nobody to meet. The filing *is* the test:
would this go in front of someone who has never seen the repository. A list built from the
project's own janitorial debt answers that question badly on the project's behalf.

Size is not a category either. An item that happens to be small is a small item in whichever file
its disposition puts it — the same reasoning [`Web/Issues/README.md`](../../Web/Issues/README.md)
already applies to the website.

## Promotion — picking a release out of `Vnext.md`

The act that turns a pool into a release. It runs once per **publish** — a minor — and never at a
patch tag, which has no ROADMAP section and no backlog. [RELEASING.md](../../RELEASING.md) step 12
is where it sits in the procedure.

1. **Read the whole file.** Two entries are only visibly the same entry, or visibly the same theme,
   with the file in front of you. A page at a time sees neither.
2. **Name the one or two themes the accumulated work has formed.** A theme is not a tag: a tag says
   what an item is about, a theme is a claim about what shipping the group would mean to somebody
   outside the project. The tags help you see a cluster; they never name it.
3. **Write the `ROADMAP.md` narrative and its success criteria — before touching the item list.**
4. **Then admit items one at a time**, each against the criterion it is meant to serve.
5. **Move the admitted ones to `BACKLOG.md`**, under bucket names matching the ROADMAP themes, which
   is the only join between those two files. Everything not admitted stays here.

**Step 3 before step 4 is the whole discipline.** Inverted, the criteria get written from the item
list, every item in the cluster passes by construction, and the admission test becomes a rubber
stamp — which is the failure that put everything into `BACKLOG.md` before this directory existed.
Written first, a criterion should reject some of the cluster that suggested it. If it rejects none,
be suspicious rather than pleased.

**A theme may instead come from the ROADMAP's own `Future` tail**, where the planned path already
lives. Both sources are legitimate and one pick can draw on both. What may not happen is a theme
with no source at all, invented while writing the narrative — that is how a release acquires goals
nobody was working toward.

**No expiry here, deliberately.** It is the counterpart to `Untriaged.md`'s ninety days rather than
an oversight: an entry in this file has already survived triage, so deleting it on a clock would
discard judgement that was actually made. The pressure valve is evidence instead of time — **an item
that several successive theme picks have passed over is making a claim about being "next" that the
record does not support**, and it moves to `Future.md`. Someone decides that, and the passes-over
are the reason.

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
| `Vnext.md` | tags | what it is and why |
| `Future.md` | tags | what it is and why |
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

**Lossy by design.** An entry that has sat in `Untriaged.md` for **ninety days** is **deleted,
unexamined** — not re-triaged. If nobody promoted it in three months it was noise, and rediscovering
it later costs less than carrying it. This rule is what stops this file becoming a second backlog,
and it is the one that will feel wrong.

**The timer is an age, and it used to be the production tag.** That worked while a cycle was a year:
an entry got a year to prove itself. A one-tag cycle carrying one or two goals can close in weeks, so
the same rule would delete a finding captured a fortnight before a release — punishing it for when it
was noticed rather than for being noise. Ninety days is the property the old rule was reaching for,
stated directly. The `@ <sha>` on every entry is what makes the age readable.

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

**Outbound, GitHub is the whole mechanism.** Work offered to a contributor is filed there and
lives there — a labelled issue can be claimed, discussed and closed against a person, none of which
a markdown file can do. What earns the label is capability the library is missing, not tidying the
library needs: [#21](https://github.com/ToddThomson/Mila/issues/21) and
[#22](https://github.com/ToddThomson/Mila/issues/22) are the shape. Beyond those,
[`BACKLOG.md`](../../BACKLOG.md) and [`ROADMAP.md`](../../ROADMAP.md) are what a contributor reads
to find the work that moves the release, and neither needs a copy kept here to be readable.
