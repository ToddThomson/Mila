# Datasets — recommended file organization

This document describes the recommended on-disk layout for datasets used by Mila samples (example: the `tiny_shakespeare` dataset). Follow this structure so sample apps (for example the Bard sample) can find the raw text, vocabulary, and optional tokenized binary datasets reliably.

## Recommended directory layout

Place each dataset in its own subdirectory under `Data/Datasets/`. For example:

```
Data/
 ├─ Datasets/
	 ├─ <dataset_name>/
		 ├─ raw/
		 │  ├─ input.txt                    # canonical raw corpus (required)
		 │  ├─ input.txt.sha256             # optional checksum
		 │  └─ README.md                    # dataset-specific notes
		 ├─ encoded/
		 │  ├─ input.txt.mila               # optional: Mila tokenized binary (fast loader)
		 │  └─ input.txt.tokens             # optional: raw token-id dump (plain binary/ints)
		 └─ tokenizers/
			 ├─ char/
			 │  ├─ input.txt.vocab           # binary char vocabulary used by CharTokenizer
			 │  └─ README.md                 # notes about char tokenizer invocation
			 └─ bpe/
				 ├─ merges.txt / vocab.json   # BPE vocabulary artifacts (if using BPE)
				 └─ tokenizer.model           # optional tokenizer model file (sentencepiece, tokenizer.json, etc.)
```
Purpose of each folder
- `raw/`
  - Store the original canonical corpus files. Never overwrite these during preprocessing; keep them as the reproducible source.
  - Samples and CLI tools typically accept a `--data-file` path that points into `raw/` (for example `Data/Datasets/tiny_shakespeare/raw/input.txt`).
- `encoded/`
  - Optional, derived artifacts — tokenized binaries for fast training (Mila-format) or simple token-id dumps.
  - Producing these speeds training (random-access, lower CPU overhead).
- `tokenizers/char` and `tokenizers/bpe`
  - Persist tokenizer outputs and assets grouped by tokenizer type.
  - `tokenizers/char` contains the binary vocabulary used by the Char tokenizer (the `.vocab` file read by `CharVocabulary::load()`).
  - `tokenizers/bpe` contains BPE merges, vocabulary JSON, sentencepiece model, or other subword artifacts.

Why this layout
- Separation of concerns: keep raw data, tokenized/encoded artifacts, and tokenizer assets isolated so:
  - Raw data is immutable and auditable.
  - Tokenizer assets are versionable and clearly associated with a particular preprocessing run.
  - Encoded files are cacheable and reused across experiments.