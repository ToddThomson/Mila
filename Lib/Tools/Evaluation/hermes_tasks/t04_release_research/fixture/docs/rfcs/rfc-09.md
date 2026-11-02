# RFC 9: Retry limit from the environment

Author: Dele Mensah
Status: accepted
Target: 1.4

Containers are configured through the environment, and today the only way to change how many times a job is
retried is `retry.limit` in `larkspur.toml`. This RFC adds `LARKSPUR_RETRY_LIMIT`, an integer, which overrides
the file when set.

The backoff between retries stays file-only; see the discussion of RFC 12 for why it is kept separate.
