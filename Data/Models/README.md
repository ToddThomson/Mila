# Models Directory

Working space for **locally converted** model weights. Nothing here is tracked in git.

This is not where installed models live. A model you install with `/install` goes into the **model
store**, which is a separate location — see below.

## The store, and why it is usually elsewhere

The store is what `/models`, `/install` and every `from_store` load read. Its root resolves in this
order (`Mila/Src/Distribution/ModelStore.ixx`):

1. `MILA_CACHE_DIR`, if set
2. `%LOCALAPPDATA%\Mila` (Windows)
3. `$XDG_CACHE_HOME/mila`, then `$HOME/.cache/mila` (Linux)

So on a normal Windows or Linux machine the store is **outside the repository**. The one place it
lands here is the container: the image sets `MILA_CACHE_DIR=/mila/Data/Models/Store`, which sits on
the repo bind mount, so a model installed from either side is the same store and survives
`run --rm`.

## What belongs in this directory

Output from `Mila/Tools/Converters/` — the fallback path, for a family Mila does not publish, a
variant it has not published, or your own fine-tune. Published models need none of this; install
them by name instead.

The converters write BF16 only. Quantized variants (FP8, FP4) are produced by Mila at load time, so
there is no quantized `.bin` to keep here. Organize by family, as the converter examples do:

```
Gpt2/gpt2_small_fp32.bin
Llama/llama_tokenizer.bin            — shared across all Llama 3.x variants
Llama/llama32_3b_instruct_bf16.bin
```

## A loose .bin is not yet a model Mila can name

Nothing loads a bare `.bin` by name. To make a converted checkpoint a first-class model — listed by
`/models`, loadable exactly like a published one — put it through the store:

```
ExportArtifact <source>.bin <artifact>.safetensors
ExportArtifact --package <dir> --weights <artifact>.safetensors --instruct
ExportArtifact --install <dir>
```

`--instruct` is not implied by the model's name; omitting it writes `instruct: false` and every
consumer then applies the wrong prompt template.

See `Mila/Tools/Converters/README.md` for converter setup, and `getting-started.md` section 5 for
the whole path from nothing to a running model.

## Notes

- Model files are large — check free space before converting.
- Original model licenses apply to converted weights; verify before any production use.
