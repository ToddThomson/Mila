# Source-model licenses

Upstream license texts for the model families Mila republishes, plus the attribution files their
licenses require. `ExportArtifact` copies them into a package with `--license` and `--notice`, and
`publish_model.py` uploads them verbatim alongside the weights.

Only families whose license imposes redistribution obligations appear here. Gemma 4 is Apache 2.0
and carries its license inside its own model card directory.

```
llama3.1/   Llama 3.1 8B Instruct and derivatives
llama3.2/   Llama 3.2 1B / 3B Instruct and derivatives
```

Each holds:

| File | Origin |
|---|---|
| `LICENSE` | `meta-llama/llama-models`, `models/llama3_{1,2}/LICENSE`, verbatim |
| `USE_POLICY.md` | the same repository, verbatim |
| `NOTICE` | generated from `LICENSE` — see below |

**3.1 and 3.2 are different agreements.** Different text, different attribution strings, different
Acceptable Use Policy URLs. A model takes the one matching its base model; they are not
interchangeable.

## Do not hand-edit these

`LICENSE` and `USE_POLICY.md` are upstream bytes. Refresh them by re-downloading, never by editing:

```
curl -sf https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE \
  -o llama3.1/LICENSE
```

`NOTICE` carries the attribution sentence the license requires be retained verbatim, and it was
extracted from `LICENSE` programmatically rather than retyped — a transcription is a chance to get
a required-verbatim string wrong. Regenerating it means extracting it again, and the check that it
is still right is that the sentence appears in `LICENSE` character for character.

## What is not here

`USE_POLICY.md` is present for reference but is **not** required to ship: clause 1.b.iv incorporates
the Acceptable Use Policy into the Agreement by reference, so distributing `LICENSE` carries it. The
`NOTICE` links it as well. Including it in a published repository is a choice, not an obligation.

The "Built with Llama" display requirement is satisfied by the model card, which is where a reader
looks; `NOTICE` repeats it.
