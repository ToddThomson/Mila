# Tokenize

Small command-line tool that preprocesses a plain text file into a vocabulary and a token stream used by the Mila data pipeline.

Overview
- Loads raw text, trains a tokenizer (via `TokenizerFactory`), builds a vocabulary and serializes:
  - `<input>.vocab` — serialized vocabulary
  - `<input>.tokens` — binary token stream

Usage
- Build and run the `Tokenize` executable, then call:
  `Tokenize <input_text_file> [--force] [--tokenizer <char|bpe>]`
- If no input file is specified the tool falls back to:
  `../Data/DataSets/TinyShakespeare/input.txt`

Options
- `--force` / `-f`  
  Force rebuild of the vocabulary and tokens even when existing .vocab and .tokens appear up-to-date.
- `--tokenizer <type>`  
  Select tokenizer type. Default: `char`. (Current implementation primarily targets the `char` tokenizer.)
- `--help` / `-h`  
  Show help message.

Output file formats
- `<input>.vocab` (binary): begins with a `size_t` representing the vocabulary size. The full vocabulary serialization format is produced by the in-repo `TokenizerVocabulary::save` implementation.
- `<input>.tokens` (binary): header is a `size_t` with the number of tokens, followed by `uint32_t` token ids (one per token). Unknown tokens are written with id `0`.

Behavior notes
- The tool will skip processing if both `.vocab` and `.tokens` exist and have modification timestamps newer or equal to the source text file, unless `--force` is used.
- Token ids are produced by the vocabulary; when a token is not found the tool writes `0` for that token id.
- The tokenizer and vocabulary implementations are provided by `TokenizerFactory` and follow the `TokenizerType` enumeration.

Examples
- Basic run:
  `Tokenize input.txt`
- Force rebuild with explicit tokenizer:
  `Tokenize input.txt --force --tokenizer char`

Limitations
- Default and tested behavior targets the `char` tokenizer. BPE or other tokenizers require respective trainer/vocabulary implementations to be present in the repository.

Contact / troubleshooting
- Errors while reading or writing files are reported to stderr and cause a non-zero exit code.
- For unexpected behavior, check file permissions and modification times of the input file and generated artifacts.
