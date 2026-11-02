# RFC 12: Backoff schedule from the environment

Author: Priya Raman
Status: accepted
Target: 1.4.1

Following RFC 9, operators asked to tune the backoff without editing `larkspur.toml`. This RFC adds
`LARKSPUR_RETRY_BACKOFF`: either a single number of seconds for a fixed delay, or `exp:<base>:<cap>` for an
exponential schedule. When set it overrides `retry.backoff` from the file.

## Open questions

- Should `0` mean "no delay" or "use the default"? (Resolved in review: no delay.)
