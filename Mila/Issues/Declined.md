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
