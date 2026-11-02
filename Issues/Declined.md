# Declined

Considered, and not doing — with the reason. Cheaper than rediscovering the argument, and the
entries here are the ones most likely to be re-proposed by someone who has not seen the
measurement.

An entry is not permanent. What changes it is new evidence, not a new opinion. Triage flow and
categories are in [README.md](README.md); the tag set is [Tags.md](Tags.md).

---

## A device-side reduction for token scoring

`perf` · `quantization` · `measured`

The model forward is 68% of scoring cost and the host transfer is negligible, so a perfect
device-side reduction is capped at **1.45x by Amdahl** — a kernel, its numerics risk and its
maintenance, for less than half a second on a run that takes minutes.

If scoring speed is ever wanted, parallelise the host `exp` loop across cores instead: the rows are
independent, no kernel, no numerics risk. Measurement in `Qwen3.8.md` §8.

## Publish the website on every push to `dev`

`docs` · `ci`

Considered 2026-10-07 after a fix sat in the repository for 29 days: `noindex` was removed from the API pages at
`0589673f` (2026-08-13), and the live site kept serving it until the next dispatch on 2026-09-11; Search Console
reported 141 pages excluded. Publishing stays a dispatch, because running `publish-site.yml` is the decision to
publish and its deploy replaces the live site wholesale; `web.yml` builds and validates the same tree on every
change without deploying. The fallout was verified gone on the live site on 2026-09-22.
