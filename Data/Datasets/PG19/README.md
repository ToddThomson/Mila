# PG-19

The long-document corpus for Gemma's quality across context lengths (`Mila/Specifications/ModelFamilyParity.md`
8.2, G2). DeepMind's PG-19: Project Gutenberg books published before 1919. The dataset is Apache 2.0 and the
texts are public domain (https://github.com/google-deepmind/pg19, read 2026-09-26).

Only the test split is used: 100 books, 41,289,101 bytes. It is not tracked; fetch it into `raw/test/`:

```bash
mkdir -p Data/Datasets/PG19/raw/test && cd Data/Datasets/PG19/raw/test
curl -s "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?prefix=test/&fields=items(name)&maxResults=1000" \
  | python -c "import json,sys; [print('https://storage.googleapis.com/deepmind-gutenberg/'+x['name']) for x in json.load(sys.stdin)['items']]" \
  | tr -d '\r' | xargs -P 8 -n 1 curl -sS -O
```
