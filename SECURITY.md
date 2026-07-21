# Security Policy

## Supported versions

Mila is pre-1.0 (currently in late alpha), and breaking changes are expected between releases.
Security fixes are applied only to the latest tagged release and the `dev` trunk. There is
no back-porting to older pre-release tags.

| Version | Supported |
|---|---|
| Latest tagged release | Yes |
| `dev` (trunk) | Yes |
| Older alpha tags | No |

## Reporting a vulnerability

Mila is maintained by a single author. Please report security concerns **privately** so they
can be addressed before public disclosure. Do not open a public GitHub issue for a security
report.

- Preferred: open a private report through GitHub's **Security Advisories** tab
  ("Report a vulnerability") at https://github.com/ToddThomson/Mila/security/advisories/new
- Alternatively: email **todd.thomson@me.com** with details and reproduction steps.

As a solo, volunteer-maintained alpha project there is no formal response-time guarantee.
You can expect a good-faith acknowledgement and, where a fix is warranted, coordination on a
disclosure timeline before any public write-up.

## Scope

Mila is an offline, source-distributed C++ inference library with no network-facing runtime
component. The most relevant concerns are around untrusted inputs to the build and load paths,
for example:

- Maliciously crafted model weight blobs parsed by the serialization / `PretrainedReader` path
- Tokenizer or configuration files supplied from an untrusted source

Reports demonstrating memory-safety or parsing issues in those paths are especially welcome.
