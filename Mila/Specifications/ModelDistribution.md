# Model Distribution

How a Mila model gets from a HuggingFace repository onto a machine and into a loader: the coordinate
that names it, the cache that holds it, and the HTTP client that fetches it.

Scoped 2026-07-29, after [ModelSerialization.md](ModelSerialization.md) Phase 7 made a 6.33 GB
pre-quantized safetensors artifact the thing worth distributing.

---

## Why

`fromPretrained()` takes a filesystem path, so using a model means already having the file. The
converter path is the only way to get one, and it needs PyTorch, 23.8 GB of source weights, and a
conversion run. That is the right workflow for adding a model family and the wrong one for using a
model Mila already publishes.

Phase 7 changed what is worth shipping: 6.33 GB, one file, format the ecosystem reads. The remaining
gap is retrieval.

---

## Goals and non-goals

**Goals.** Name a published model with one string. Fetch it once, verify it, reuse it. Keep local
files working exactly as they do today. Support gated repositories, because Llama is gated.

**Non-goals, and they are firm:**

- **Mila does not run a registry.** HuggingFace is the registry. The `mila-llm` organization on the
  ToddThomson account is the namespace.
- **Mila does not read another tool's cache.** Not ollama's content-addressed store, not LM Studio's
  repo mirror. Interoperating with either is out of scope permanently.
- **Mila does not upload.** Publishing is a human action through HuggingFace's own tooling. Nothing in
  the library writes to a remote.
- **No model assets in the `mila-llm` wheel.** A 6.33 GB wheel is unshippable and the decision is
  settled; see [PythonBinding.md](PythonBinding.md).

---

## The coordinate

```
<organization>/<repository>:<variant>
```

For example `mila-llm/gemma-4-12b-it:fp4`. Variant is separate from repository because variants share
components: FP4, FP8 and BF16 of one model share a 14 MB tokenizer, and a flat
`<repository>-<variant>` name makes that sharing invisible.

An optional `hf:` prefix forces coordinate interpretation. A bare `<organization>/<repository>` with
no variant resolves to the manifest's declared default.

**Local paths stay first-class.** A spec that names an existing file is used as-is, with no network
access and no cache involvement. This is the lesson from ollama, whose content-addressed store cannot
consume a file the user already has. Mila's resolver takes a path or a coordinate, and the model
catalog holds either.

---

## The repository manifest

Each published repository carries `mila.json` at its root:

```json
{
  "manifest_version": 1,
  "architecture": "gemma",
  "default_variant": "fp4",
  "variants": {
    "fp4": {
      "minimum_mila_version": "0.20.0",
      "weight_quantization": "per_group_fp4_128",
      "files": {
        "weights":   { "path": "gemma4_12b_it_fp4.safetensors", "sha256": "d49c…", "bytes": 6799927760 },
        "tokenizer": { "path": "gemma_tokenizer.bin",           "sha256": "…",     "bytes": 14198878 }
      }
    }
  }
}
```

HuggingFace's own API already reports file listings and LFS digests, so a manifest is not needed to
discover *what is there*. It is needed for what the API cannot know: which files compose a loadable
model, which variant a caller means, what quantization the bytes carry, and the oldest Mila that can
read them. One small GET buys all of that and makes the repository self-describing.

`minimum_mila_version` is the version-skew guard. A newer artifact loaded by an older Mila fails with
a version comparison rather than a parse error somewhere inside the tensor index.

---

## The cache

Content-addressed, which is the one thing worth taking from ollama's design. The digest arrives with
the manifest and HuggingFace serves it as the ETag on LFS files, so it costs nothing to key on.

```
<cache-root>/
  manifests/<organization>/<repository>/<variant>.json   resolved manifest as fetched
  blobs/sha256-<hex>                                     the files themselves
  tmp/<unique>                                           in-flight downloads only
```

Root resolution, first match wins:

1. `MILA_CACHE_DIR` if set
2. `%LOCALAPPDATA%\Mila\models` on Windows
3. `$XDG_CACHE_HOME/mila/models`, else `~/.cache/mila/models`

Four properties follow from content-addressing, and each replaces code that would otherwise be
written by hand:

| Property | Why it is free |
|---|---|
| Integrity | The path is the digest; a blob in place has been verified |
| Deduplication | Variants sharing a tokenizer share its blob |
| Resume | Download to `tmp/`, verify, rename into `blobs/` |
| Atomicity | A blob is either absent or complete; a torn write never has a valid name |

A partially written blob never occupies its final path, so an interrupted download cannot be mistaken
for a good one. That is the failure the design is chosen to make impossible rather than to detect.

---

## Retrieval

```
resolve(spec) -> ResolvedModel { weights_path, tokenizer_path, manifest }
```

1. If `spec` names an existing file, return it directly. No network, no cache.
2. Parse as a coordinate; reject anything that is neither.
3. Fetch `mila.json`, select the variant, check `minimum_mila_version`.
4. For each file: if `blobs/sha256-<digest>` exists, use it. Otherwise download to `tmp/`, hashing as
   the bytes arrive, then verify and rename.
5. Return the blob paths.

Hashing during the transfer rather than in a second pass matters at 6.33 GB: a verification re-read
would double the I/O for no additional confidence.

### Authentication

Gated repositories require a token. **Llama 3.2 and 3.1 are gated**, as
`Tools/Converters/README.md` already documents; Gemma 4 under Apache 2.0 is not (see
[project memory on the license change](ModelSerialization.md) and NOTICE.md).

Token discovery, first match wins: `MILA_HF_TOKEN`, `HF_TOKEN`, then
`~/.cache/huggingface/token` (where `huggingface-cli login` writes it).

Two failures that need different messages, because conflating them wastes an afternoon:

- **401** — no token, or the token is invalid. Say how to obtain one.
- **403** — the token is valid but the repository's terms have not been accepted. Name the model page
  to accept on.

### The redirect is a security boundary

HuggingFace redirects an LFS file to a pre-signed URL on a CDN host. An `Authorization` header set
through `CURLOPT_HTTPHEADER` **is** forwarded across a cross-host redirect, so following redirects
automatically leaks the token to the CDN.

**Do not enable `CURLOPT_FOLLOWLOCATION`.** Read the `Location`, then re-issue without the auth header
when the host changes. The pre-signed URL carries its own authorization and needs no token.

### Resume

`Range: bytes=<offset>-` against the existing `tmp/` length. A **206** resumes. A **200** means the
server ignored the range and is sending the whole file, so the partial must be discarded and the hash
restarted -- treating a 200 as a resume silently concatenates and produces a corrupt blob that only
the final digest check catches.

---

## The HTTP client

libcurl, one implementation for both platforms. Windows has no linkable OS libcurl, so it is vendored
there regardless; vendoring on Linux too buys one known version everywhere instead of whatever the
distribution shipped, matching how nlohmann, cutlass and pybind11 are already pinned.

- Pinned tag through CPM, built static -- no runtime DLL, and the CPM consumer gate stays clean
- **Windows: Schannel.** OS-provided TLS, Windows certificate store, no CA bundle to ship or refresh
- **Linux: system OpenSSL.** Also the platform trust store
- Protocols reduced to HTTP and HTTPS. No LDAP/LDAPS, telnet, dict, gopher, SMTP/POP3/IMAP, RTSP,
  TFTP, MQTT, SMB, FTP
- No nghttp2, brotli or zstd: HuggingFace serves LFS blobs uncompressed over HTTP/1.1, so HTTP/2 and
  content encodings add dependencies and buy nothing

### `MILA_ENABLE_MODEL_DOWNLOAD`

Defaults ON, matching `MILA_ENABLE_CUDA` and `MILA_ENABLE_PYTHON_BINDINGS`. Two reasons it exists:

**libcurl and libssl are not on the manylinux whitelist**, so a Linux `mila-llm` wheel cannot link
them. The wheel does not need to: Python already has `huggingface_hub`. The wheel builds with the
feature off, resolves a path in Python, and passes it to Mila -- which the resolver already supports,
because local paths are first-class. The C++ client exists for Chat and native consumers, which is
where it is wanted.

Second, a library whose entire third-party surface is two headers should not force a network
dependency on a consumer that only loads from disk.

---

## Licensing per family

The publishing story is not uniform, and assuming it is would be the first mistake.

| Family | License | Republishable to `mila-llm` |
|---|---|---|
| Gemma 4 | Apache 2.0 | Yes, public and ungated |
| Gemma 3 and earlier | Gemma Terms of Use | Terms propagate; gate to match Google |
| Llama 3.1 / 3.2 | Llama Community License | Propagates; Meta gates the source |

Gemma 4's Apache 2.0 still requires the license text, attribution, and a statement that changes were
made -- quantization is a modification. That belongs in the repository alongside the root NOTICE.md
habit already established.

For a family that cannot be cleanly republished, the coordinate must fail usefully rather than 403 into
a wall: the manifest can declare a variant unavailable and name the conversion path instead.

---

## Build plan

**Phase 1 -- the client.** CPM wiring, `MILA_ENABLE_MODEL_DOWNLOAD`, and an HTTP module: GET with
progress, manual redirect handling, `Range` resume, token injection, 401/403 discrimination.
*Done when:* a small public file downloads, resumes correctly from a truncated partial, and a 200
response to a range request restarts rather than concatenates.

**Phase 2 -- the cache.** Root resolution, blob store, `tmp/` staging, hash-during-transfer, atomic
rename.
*Done when:* the same blob fetched twice hits the cache on the second call, and an interrupted
transfer leaves nothing in `blobs/`.

**Phase 3 -- the resolver.** Coordinate parsing, manifest fetch and validation, variant selection,
version-skew check, path-or-coordinate dispatch.
*Done when:* a coordinate and a local path both load the same model, and a bumped
`minimum_mila_version` refuses.

**Phase 4 -- the catalog.** Chat entries carry a coordinate or a path. `gemma-12b-packed` becomes
`mila-llm/gemma-4-12b-it:fp4`.
*Done when:* a clean machine runs `/model gemma-12b-packed` and gets a coherent Gemma 4 session with
no manual download.

**Phase 5 -- publish.** The `mila-llm/gemma-4-12b-it` repository: artifact, tokenizer, `mila.json`,
Apache 2.0 license text, attribution, modification statement.

---

## Open decisions

1. **Revision pinning.** A coordinate resolves against `main` today. A `@<revision>` suffix would make
   a load reproducible against a moving repository. Decide before the first published artifact is
   updated, not after.
2. **Whether the manifest is fetched every load.** Caching it makes startup offline-capable and makes
   a republished artifact invisible until the cache is cleared. An ETag revalidation is the middle
   path and costs one conditional GET.
3. **Progress reporting surface.** Chat wants a rendered bar; a library callback taking bytes-so-far
   and total is the minimum. Whether anything else consumes it is unsettled.
