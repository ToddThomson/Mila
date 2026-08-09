# Publishing

`publish_model.py` uploads a prepared model to the HuggingFace Hub. It is the only thing in Mila that
uploads — the library never does, and `ExportArtifact` never does.

It validates every declared file against `mila.json` before the first byte goes out, verifies the
repository listing afterwards, and skips anything the Hub already holds at the same digest. Re-running
it is the normal way to finish an interrupted upload.

```
python publish_model.py <directory> [--repo <owner>/<name>] [--dry-run]
```

## The two directory shapes

Both are accepted, and the difference is only where the multi-gigabyte files live.

**A package directory** — built by `ExportArtifact --package`, holds every file it declares. Needs
`--repo`, because it carries no `publish.json`. This is the shape to use.

**A card directory** — `Mila/Tools/ExportArtifact/ModelCards/<name>/`, holds the small files and a
`publish.json` mapping Hub paths to weights kept outside the repository. `gemma-4-12b-it-fp4` was
published this way. Mapped paths resolve against the Mila repository root, which the script derives
as four parents above the card directory, so a card directory only works at its established depth.

## Preparing a model

1. **Convert** the upstream checkpoint with `Tools/Converters/<Family>/`. Converters always write
   BF16.

2. **Quantize** to the published form. Publishing is FP4 only:

   ```
   ExportArtifact <source.bin> <artifact.safetensors> --quantization fp4
   ```

3. **Write the model card** as `ModelCards/<name>/README.md`. It carries the HuggingFace YAML front
   matter (`license`, `base_model`, `tags`, `library_name: mila`), states that the artifact is
   loadable only by Mila and deliberately not NVFP4 or MXFP4, and lists the modifications from the
   base model. The two Llama cards are the template.

4. **Assemble the package.** The name given here is the published coordinate and is permanent once
   anything links to it:

   ```
   ExportArtifact --package <package-dir> --weights <artifact.safetensors> \
     --tokenizer <tokenizer.bin> \
     --license  Licenses/<family>/LICENSE \
     --notice   Licenses/<family>/NOTICE \
     --model-card ModelCards/<name>/README.md \
     --base-model <upstream-repo-id> \
     --license-id <license-id> \
     --as <name>
   ```

   `--notice` applies only to families whose license requires attribution in a file of its own; Llama
   does, Gemma does not. See `ExportArtifact/Licenses/README.md` — 3.1 and 3.2 are separate
   agreements and are not interchangeable.

5. **Install and run it** before publishing. Load-time coherence is not implied by a digest match:

   ```
   ExportArtifact --install <package-dir> --keep
   ```

   `--keep` copies rather than moves. **Without it the package directory is consumed by the install
   and there is nothing left to upload.**

## Publishing

```
python publish_model.py <package-dir> --repo mila-llm/<name> --dry-run
python publish_model.py <package-dir> --repo mila-llm/<name>
```

The repository does not need to exist first. A new one is **created private, filled, verified, and
only then made public**, so a model is never reachable in a half-uploaded state — the weights are
the last and longest upload, and a public-from-the-start repository would advertise a model that
cannot be loaded for as long as that takes. If verification fails the repository stays private and
the run can simply be repeated.

A repository that already exists keeps whatever visibility it has. Taking a live model private to
replace one small file would be an outage caused by the tool meant to maintain it.

The dry run hashes every declared file and reports what would be uploaded. It is worth running on its
own: a manifest that disagrees with the bytes produces a repository that fails verification on every
download, and hashing 5 GB is cheaper than discovering that after the transfer.

Small files go first — `mila.json`, `README.md`, `LICENSE`, `NOTICE` — so a mistake in them is visible
before a multi-gigabyte upload begins.

A missing `NOTICE` is reported when the license id begins with `llama` and passes silently otherwise,
which is why the license id in the manifest is load-bearing rather than decorative.

## Authentication

`HF_TOKEN`, `MILA_HF_TOKEN`, or a stored `hf auth login` token, in that order — whichever
`huggingface_hub` resolves. The token needs write access to the target repository and is never
printed. Publishing under `mila-llm` requires membership in the organization.

## Environment

Run from the converter virtual environment, which is where `huggingface_hub` is installed:

```
Mila/Tools/Converters/.venv/Scripts/python.exe -m pip install -U huggingface_hub
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Published, or a dry run completed |
| 2 | Validation failed — nothing was uploaded |
| 3 | Not authenticated |
| 4 | A file was missing from the repository after upload |
