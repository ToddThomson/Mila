# tiny_shakespeare dataset — organization and preprocessing for Bard

This document explains the on-disk layout expected by the Bard sample and how to preprocess the Tiny Shakespeare text using the Mila tokenizer tool (tokenizer.exe) so the Bard sample can load the data and vocabulary.

## Directory layout (expected)
Place the dataset under `Data/Datasets/tiny_shakespeare/` with the following recommended files:

- `input.txt`
  - Raw text corpus (one file containing the training text). This is the path the Bard sample expects by default.
- `input.txt.vocab`
  - Binary vocabulary file produced by the Mila tokenizer. Bard looks for `<data_file>.vocab` (where `data_file` is the path you pass to the sample).
- `input.txt.mila` (optional)
  - Tokenized binary dataset (optional). The tokenized format uses the Mila tokenized file header (magic `"MILA"`, version, tokenizer type) and stores token ids; having this speeds training dataset loading and random-access reads.

Note: filenames above are examples. The Bard sample uses the `--data-file` path you pass; it expects a matching `.vocab` file at `<data-file>.vocab`.

## What to generate with the tokenizer
Run the Mila tokenizer to create:
- The vocabulary binary (`.vocab`) that `CharVocabulary::load()` reads.
- (Optional) A tokenized dataset file (`.mila` / token id binary) containing token ids and the Mila tokenized header for faster dataset reading.

The Bard sample requires at minimum the `.vocab` file to be present. If only raw `input.txt` and `input.txt.vocab` exist, the data loader may perform in-memory encoding as needed; providing a tokenized binary file improves throughput.

## Example tokenizer usage (recommended)
The exact CLI may vary depending on the tokenizer build in your tree. The examples below show a typical invocation and flag meanings — adjust flags if your tokenizer binary uses different names.

Windows (cmd/powershell):
tokenizer.exe --input Data\Datasets\tiny_shakespeare\input.txt 
--vocab-out Data\Datasets\tiny_shakespeare\input.txt.vocab 
--tokens-out Data\Datasets\tiny_shakespeare\input.txt.mila 
--tokenizer-type char 
--case-insensitive 
--use-pad --use-unk

Linux/macOS:

./tokenizer --input Data/Datasets/tiny_shakespeare/input.txt 
--vocab-out Data/Datasets/tiny_shakespeare/input.txt.vocab 
--tokens-out Data/Datasets/tiny_shakespeare/input.txt.mila 
--tokenizer-type char 
--case-insensitive 
--use-pad --use-unk

Common flags explained:
- `--input` — path to raw text (`input.txt`).
- `--vocab-out` — where to write the binary vocabulary file (`<input>.vocab`).
- `--tokens-out` — (optional) write token ids in Mila tokenized format for faster loading.
- `--tokenizer-type` — `char` for the Bard pipeline (single-byte character mapping).
- `--case-insensitive` / `--case-sensitive` — whether to lowercase bytes when building vocab.
- `--use-pad` / `--use-unk` — include PAD/UNK special tokens in the vocabulary (recommended).

If your project includes a tokenizer implementation under `Samples/Bard/Tools` or a build target that produces `tokenizer.exe`, build that target first (via the usual CMake/Ninja build process) and run the produced binary.

## Running the Bard sample
After producing `input.txt.vocab` (and optionally the tokenized `.mila` file), build the Bard sample and run:
Bard.exe --data-file Data\Datasets\tiny_shakespeare\input.txt 
--batch-size 32 --seq-length 128 --epochs 10


Important:
- The sample will look for the vocabulary at `Data\Datasets\tiny_shakespeare\input.txt.vocab` (it appends `.vocab` to the `--data-file` path).
- If you created a tokenized binary (`.mila`), ensure the dataset reader/loader for Bard is configured to use it; otherwise the loader will fall back to encoding/reading from the raw text using the `.vocab`.

## Tips and troubleshooting
- Ensure the tokenizer version and the CharVocabulary format match; the vocabulary binary format used by `CharVocabulary::save()` / `CharVocabulary::load()` must be compatible with the tokenizer tool you run.
- If you see missing-token errors, verify whether you included an UNK special token when generating the vocabulary. If the vocab lacks UNK and the corpus contains bytes outside the vocab, encoding will fail or replace with fallback behavior.
- For large corpora, prefer writing a tokenized `.mila` file (token ids + header) — the dataset reader is optimized for random access and streaming.
- The tokenized file uses the Mila tokenized file header (`TOKENIZED_FILE_MAGIC_NUMBER = "MILA"`) so readers can validate format and version.

If you want, I can:
- Add a small helper script (python) that invokes the tokenizer and validates outputs, or
- Inspect the repository for the exact tokenizer CLI and produce a precise command line for your workspace.