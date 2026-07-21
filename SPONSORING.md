# Sponsoring Mila

Mila accepts **compute and hardware sponsorship only.**

Mila is an independent, from-scratch C++23 / CUDA inference runtime for LLMs, built to understand
these models at the metal. It is neither vLLM nor llama.cpp — a separate implementation of the same
math — which makes it a useful second opinion: on Mila's path, a vendor can tell a *hardware* bug
from a *framework* bug. Native FP4 on Blackwell sits exactly on that seam.

## What would help

Access to hardware Mila cannot currently reach for validation:

- **Large-VRAM GPUs (80–96 GB)** — long-context validation beyond the 12 GB consumer ceiling.
- **Multi-GPU** — tensor-parallel paths that are untestable on a single card today.
- **Blackwell (sm_120)** — native FP4 on current silicon (partly self-funded via an RTX 5060 Ti 16GB).

This is a **marketing exchange, not philanthropy**: a runtime that runs well on your hardware — and
that can isolate your hardware's behaviour independently of the large frameworks — is a validation
surface for it.

## What Mila does *not* accept

Money, in any form — no GitHub Sponsors, no cash tiers, no donations. This is a craft project; the
only useful currency here is compute.

## Getting in touch

For compute or hardware collaboration, contact **todd.thomson@me.com**.

Compute sponsors are acknowledged, with thanks, once they come aboard.
