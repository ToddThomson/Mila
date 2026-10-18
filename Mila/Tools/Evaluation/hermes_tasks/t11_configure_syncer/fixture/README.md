# syncer

Copies new and changed files from one directory into an archive directory. When a file changes, the archived
copy is kept as `name.1`, the one before that as `name.2`, and so on.

## Configuration

syncer reads `syncer.ini` from the directory it is run in. Everything is in the `[sync]` section:

| Setting | Meaning |
|---|---|
| `source` | Directory to copy from. Required. |
| `destination` | Archive directory. Created if missing. Required. |
| `exclude` | Comma-separated file name patterns not to copy, e.g. `*.bak, ~*` |
| `keep_versions` | How many versions of each file to keep in the archive, counting the current one. Default 1. |

`syncer.ini.example` shows a typical setup.

Run `python3 syncer.py --check` to validate the configuration without copying anything.
