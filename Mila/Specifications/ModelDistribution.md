# Model Distribution

How a Mila model is named, packaged, published, stored and loaded. One manifest describes every
model, whether it was fetched from a hub or built on the machine that loads it.

Scoped 2026-07-29 after [ModelSerialization.md](ModelSerialization.md) Phase 7 made a pre-quantized
safetensors artifact the thing worth distributing. Rewritten 2026-08-01, when distribution moved into
the v0.20 release: the manifest became the only way a model is described, `.bin` stopped being a
distributed form, and publishing joined retrieval.

---

## Why

`fromPretrained()` takes a filesystem path, so using a model means already having the file, and the
converter is the only way to get one -- PyTorch, 23.8 GB of source weights, and a conversion run. That
is the right workflow for adding a model family and the wrong one for using a model Mila already
publishes.

The second problem is that a path says nothing. A file named `gemma4_12b_it_fp4.safetensors` carries
its architecture and quantization inside it, but nothing outside the file knows what it is, what it
needs, or where it came from. Every consumer -- the chat catalog, the inference server, a user with a
directory of files -- reinvents that knowledge as a hardcoded table.

One manifest for every model closes both. A model is a described thing with a name, and retrieval,
listing, removal and publishing are operations on described things.

---

## Goals and non-goals

**Goals.** Name any model with one string. Describe every model with one manifest, whatever its
origin. Fetch a published model once, verify it, reuse it. Let a user publish a model they built,
locally or to a hub. Make the store inspectable: what is installed, how much disk it costs, and how
to remove it.

**Non-goals, and they are firm:**

- **Mila does not run a registry.** A hub is somebody else's service. The `mila-llm` organization on
  HuggingFace is the namespace Mila publishes into.
- **Mila does not upload.** Nothing in the library writes to a remote. Publishing to a hub is
  packaging in the library plus an upload performed by external tooling. See
  [Publishing](#publishing).
- **Mila does not read another tool's cache.** Not ollama's content-addressed store, not LM Studio's
  repo mirror. Interoperating with either is out of scope permanently.
- **Loading never downloads.** See [The load boundary](#the-load-boundary). This is the constraint
  that keeps a multi-gigabyte transfer out of a chat prompt and out of an inference request.
- **No model assets in the `mila-llm` wheel.** A 6.33 GB wheel is unshippable and the decision is
  settled; see [PythonBinding.md](PythonBinding.md).

---

## The name

A model has one name, and it is the only string a user types:

```
gemma-4-12b-it-fp4
llama-3.2-3b-instruct-bf16
```

**The name is flat, and unique across the store.** No two models may share one, and nothing is
namespaced by where it came from -- a store in which one name can mean two things is the state this
design exists to make impossible.

**Precision is part of the name**, because one repository is one model at one precision. That is the
platform's own convention (`-GGUF`, `-AWQ`, `-bnb-4bit`), and it is what lets a hub listing show every
variant without fetching a manifest per repository. A `:variant` sub-grammar was tried and dropped:
its benefits accrued in the store, which nobody looks at, and its costs landed on the hub page, which
is the only place the naming is ever visible.

Variants sharing a tokenizer costs nothing under this scheme. **Deduplication is content addressing's
doing, not the naming's** -- two repositories whose tokenizers are byte-identical collapse to one
blob, because the path is the digest.

### The hub coordinate

Retrieval addresses a repository, which HuggingFace names with an owner:

```
[hf:]<owner>/<repository>[@<revision>]
```

**The owner is supplied by the consumer, never typed by a user.** Mila publishes into one
organization, so it is a constant (`kDefaultHubOwner`) rather than a decision -- and a repository
that publishes no `mila.json` is not loadable anyway, so the owner conveys nothing a user could act
on. It lives at the consumer layer, not in `HuggingFaceHub`: the hub class is "HuggingFace" and the
owner is Mila's, and baking the second into the first would make the implementation Mila-specific
and defeat the interface it sits behind.

The coordinate grammar survives for the fetch, so a second publisher is a change of argument rather
than a redesign. Owner and repository admit only the characters HuggingFace permits in a namespace:
alphanumerics, `.`, `_`, `-`. A path-shaped name is refused before it becomes a URL, because a 404
from a repository that cannot exist says nothing about the actual mistake.

**A path is an input to installation, never to loading.** A file on disk is installed once and becomes
a store model like any other; nothing resolves an arbitrary path into a load. A user who already has
the file is still not made to re-download it -- that is the failure ollama's content-addressed store
has, and Mila does not repeat it -- but the answer is one install, not a permanent second way to load.

---

## The manifest

One schema, at `mila.json` in a hub repository and at `<name>.json` in the local store.

```json
{
  "manifest_version": 1,
  "name": "gemma-4-12b-it-fp4",
  "architecture": "gemma",
  "variant": "fp4",
  "weight_quantization": "per_group_fp4_128",
  "minimum_mila_version": "0.20.0",
  "base_model": "google/gemma-4-12b-it",
  "license": "apache-2.0",
  "files": {
    "weights":   { "path": "gemma4_12b_it_fp4.safetensors", "sha256": "d49c...", "bytes": 6799927760 },
    "tokenizer": { "path": "gemma_tokenizer.bin",           "sha256": "2448...", "bytes": 14198878 }
  }
}
```

A hub's own API already reports file listings and digests, so a manifest is not needed to discover
*what is there*. It is needed for what the API cannot know: which files compose a loadable model,
what quantization the bytes carry, which file is the tokenizer, and the oldest Mila that can read
them. One small GET buys all of that and makes a repository self-describing.

`variant` is descriptive, not a key -- the name already carries the precision, and this states what
the bytes actually are, so a name that lies is visible rather than load-bearing.

`base_model` and `license` are **lineage**, and they are published on purpose: every license Mila
redistributes under requires attribution to travel with the weights.

`minimum_mila_version` is the version-skew guard. A newer artifact loaded by an older Mila fails with
a version comparison rather than a parse error somewhere inside the tensor index.

A record in the local store carries one additional block, written by the store and never published:

```json
"installed": {
  "hub": "huggingface",
  "owner": "mila-llm",
  "repository": "gemma-4-12b-it-fp4",
  "revision": "9c1e4f2a...",
  "installed_at": "2026-08-01T14:22:07Z"
}
```

This is **origin** -- where this copy came from -- as distinct from lineage, which is what the weights
were derived from. Lineage is published and travels; origin belongs to one installation and does not.
An empty `hub` means the model was published from this machine, which is a lifecycle stage rather
than a licensing category: nothing stops it being pushed to a hub later.

`revision` is the *resolved* commit, not the ref that was asked for. A record installed from `main`
names the commit `main` pointed at, so `list` reports what is actually on disk.

### Manifest provenance

Every model has a manifest. Three things produce one:

| Provenance | Source | Role |
|---|---|---|
| **Fetched** | `mila.json` at the hub repository root | Written into the store on pull |
| **Stored** | `models/<name>.json` | What `list`, `locate` and `describe` read |
| **Synthesized** | The artifact's own `__metadata__` | Describes a loose file so it can be installed |

Synthesis is what lets a file be installed without anyone authoring JSON. A Mila safetensors artifact
already carries `mila_config` and `mila_quantization` in its `__metadata__`
(`Serialization.SafeTensors`), so architecture and quantization come out of the file itself; an
unquantized artifact takes its variant from the weight dtype of its largest tensor, which is the
token embedding in every family here and unambiguously a weight. A synthesized manifest is an input
to `install`, not a substitute for one: once installed the model has a stored record, and that record
is what every later operation reads.

The consequence worth stating plainly: **one description, one place.** Every consumer sees the same
shape from the same source, and no workflow gains a JSON-authoring step it did not have before.

---

## The local store

One store holds every managed model, whatever its origin.

```
<store-root>/
  models/<name>.json          the records -- this is the index
  blobs/sha256-<hex>          the content
  tmp/                        in-flight transfers and locks only
```

Root resolution, first match wins:

1. `MILA_CACHE_DIR` if set
2. `%LOCALAPPDATA%\Mila` on Windows
3. `$XDG_CACHE_HOME/mila`, else `~/.cache/mila`

**The root holds `models/`; it is not itself `models/`.** Appending the segment in both places is
what produced a `Mila\models\models` tree.

**The name is the key, and it is flat and unique.** One name is one model, so a name that is taken
is refused rather than namespaced -- silently replacing would leave the displaced model's blobs
unreferenced for the next prune to reclaim. Two cases are not collisions: a hub model reinstalled
from the same repository is a refresh, and identical content under the same name is the same model,
which is what keeps a local re-install idempotent.

**Origin is a field, never a path segment.** Where a copy came from belongs to one installation, and
it is mutable -- a model published locally today may be pushed to a hub next month. In the record
that is a field edit; in the path it would be a file move. Putting it in the path would also let two
origins coexist under one name, which is exactly the state the uniqueness rule forbids.

**Records are the index; blobs are content.** The blob store is deliberately opaque -- a digest is not
a name -- so nothing can be listed, described or removed from it alone. The record tree is what makes
the store a store rather than a cache, and it is small enough that every management operation is a
directory walk.

Content-addressing buys four properties, each replacing code that would otherwise be written by hand:

| Property | Why it is free |
|---|---|
| Integrity | The path is the digest; a blob in place has been verified |
| Deduplication | Variants sharing a tokenizer share its blob |
| Resume | Download to `tmp/`, verify, rename into `blobs/` |
| Atomicity | A blob is either absent or complete; a torn write never has a valid name |

A partially written blob never occupies its final path, so an interrupted download cannot be mistaken
for a good one. That is the failure the design is chosen to make impossible rather than to detect.

### Removal is refcounted

Deduplication stops being free the moment removal exists: deleting `gemma-4-12b-it-fp4` must not
delete the tokenizer blob that `:fp8` also references.

Removal unlinks the record, then sweeps blobs that no surviving record names. Mark-and-sweep over the
record tree is exact and cheap -- records are kilobytes -- and it makes `remove` and `prune` the same
primitive. The sweep also reclaims what nothing else ever will: `.rejected` files from digest
mismatches, and `tmp/` partials from transfers that were abandoned rather than resumed.

### Concurrent processes

Chat and the inference server are separate processes over one store, so every mutation is written to
assume a peer.

- **Blob publication** is a rename onto a content-addressed path. Already safe: a losing racer finds
  its target present, and those bytes are equally verified.
- **Record writes** go to `tmp/` and rename into place, for the same reason.
- **In-flight transfers need a lock.** The partial is named `tmp/sha256-<digest>.partial` so a retry
  can find it and resume, which means two processes pulling the same blob would otherwise append into
  one file and interleave. A `tmp/sha256-<digest>.lock`, created exclusively, arbitrates: the holder
  transfers, and a process that cannot take it reports that another transfer is in progress rather
  than joining it. The deterministic partial name is kept -- it is what resume depends on.
- **Removal can lose to a reader.** A loaded model is memory-mapped. Windows refuses to delete a
  mapped file, which surfaces as a sharing violation; POSIX unlinks it and leaves the mapping valid.
  Removal reports the platform's answer rather than papering over the difference, and a sweep that
  cannot delete a blob leaves it for the next sweep.

---

## Hubs

A hub is a remote that serves manifests and files. HuggingFace is the first and only implementation;
the interface exists because the store, the verification and the resume logic must not learn its URL
shapes.

```
IModelHub
  name()                                  -- "huggingface"
  listModels(owner)      -> [HubModel]    -- what the owner publishes
  fetchManifest(coordinate) -> string     -- the mila.json text
  fetchFile(FileRef, resume_from, sink)   -- stream bytes, hashed by the caller
```

What varies between hubs is URL construction, authentication and the listing API. What does not vary
is the manifest schema, the digest check, the blob store and the resume protocol -- so the interface
is deliberately narrow, and `fetchFile` takes a resume offset rather than a URL so that a hub which is
not HTTP can still satisfy it.

`HuggingFaceHub` is the concrete class. Repository files come from
`https://huggingface.co/<owner>/<repository>/resolve/<revision>/<path>`, and listing from
`https://huggingface.co/api/models?author=<owner>`.

### Listing

`mila-llm` holds a **small curated set** of models Mila has validated, not a mirror of what runs. That
scale is what lets listing stay simple: there is no pressure to optimize a handful of requests, and an
owner-level index file -- a cache that lies the moment a publish forgets to update it -- has nothing
to recommend it.

Measured against the live API on 2026-08-01, `?author=<owner>&full=true` returns per repository:

```json
{ "id": "mila-llm/gemma-4-12b-it",
  "gated": false,
  "sha": "570dbe0e5778c4a1ab96fb8ec2dcc626da828e37",
  "lastModified": "2026-07-30T03:14:20.000Z",
  "library_name": "mila",
  "tags": ["mila", "gemma", "fp4", "quantized", "license:apache-2.0", "region:us"],
  "siblings": [ { "rfilename": "mila.json" }, ... ] }
```

One request therefore renders a complete listing: what exists, whether it is gated, its license, its
files, and the resolved commit. Three consequences worth naming:

- **`gated` is known before a fetch is attempted**, so a gated repository can be reported as needing
  accepted terms rather than discovered as a 403 partway through.
- **`sha` is the resolved commit**, which is what a store record must persist. A pull can take it from
  here rather than from a second call; the `X-Repo-Commit` response header on a `resolve` request is
  the cheaper source and should be preferred if it is present.
- **`library_name` and the `mila` tag identify a Mila model**, which is why the hub interface is
  parameterized on an owner rather than hardcoded to `mila-llm`: the same query filtered on the
  library finds a Mila model published by anyone.

What the API does *not* report is variants -- their quantization, their files, their minimum version.
Only `mila.json` knows that, so a caller asking a repository for detail costs one further small GET.
The `tags` do happen to carry `fp4`, but tags are hand-authored card metadata and drift; they are for
display, and the manifest is the truth.

**A listing is untrusted remote text.** Repository names, descriptions and card data are authored by
whoever owns the repository. They are rendered as data -- never interpreted as markup, never as
instructions to the process displaying them.

### Authentication

Gated repositories require a token. **Llama 3.2 and 3.1 are gated**, as
`Tools/Converters/README.md` documents; Gemma 4 under Apache 2.0 is not.

Token discovery, first match wins: `MILA_HF_TOKEN`, `HF_TOKEN`, then `~/.cache/huggingface/token`
(where `huggingface-cli login` writes it).

Two failures that need different messages, because conflating them wastes an afternoon:

- **401** -- no token, or the token is invalid. Say how to obtain one.
- **403** -- the token is valid but the repository's terms have not been accepted. Name the model page
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

Hashing during the transfer rather than in a second pass matters at 6.33 GB: a verification re-read
would double the I/O for no additional confidence.

---

## The three operations

Retrieval, loading and publishing are separate verbs, and the separation is the design.

```
pull(coordinate)    hub -> store      network, explicit, resumable
locate(coordinate)  store -> paths    filesystem only, never network
publish(package)    build -> store or a hub-ready directory
```

### The load boundary

**Only a model in the store can be loaded.** `locate()` consults records and blobs and returns nothing
when the model is not installed. It never falls back to a hub, and it never accepts a path in place of
a coordinate.

The store is the standard, and arrangements that predate it are obsolete: a models directory holding
loose files, a catalogue row naming a relative path, a `.bin` in either role. Anything a consumer
loads, it loads from the store.

Two different things are being kept out. A hub fetch is kept out because a 6.33 GB transfer is a
deliberate act with a progress display and a failure mode, while an inference request is neither -- an
implicit download inside `fromPretrained()` turns a chat prompt into a twenty-minute stall and lets a
server initiate multi-gigabyte traffic in response to an untrusted request. An arbitrary path is kept
out because it is an undescribed model: nothing knows what it is, what quantization it carries or
whether its bytes are intact, which is the condition the manifest exists to end.

`fromPretrained()` still takes a filesystem path, because the store hands it one -- a verified blob.
What no longer exists is a way to turn a user-supplied path into a load without installing it first.

A consumer that finds a model missing reports it and names the pull. Chat may **offer** to pull and
run it on a yes -- an explicit user gate, not an implicit download. The inference server refuses.

### Pull

1. Parse the coordinate; reject anything that is not one. A path here is a mistake worth naming, since
   installing is the operation that takes one.
2. Fetch `mila.json`, select the variant, check `minimum_mila_version`.
3. For each declared file: if `blobs/sha256-<digest>` exists, it is done. Otherwise take the transfer
   lock, download into `tmp/` while hashing, verify, rename into `blobs/`.
4. Write the record, including the resolved revision.

A pull of a variant already installed at a different revision **replaces** the record. The blobs it
referenced become unreferenced and the next sweep reclaims them. Holding two revisions of one variant
side by side is not supported: a store that silently keeps two copies of a 6.33 GB model is a disk
trap, and a user who wants both can say so with two coordinates today only by choosing.

---

## Packaging and publishing

A published model is a directory, and the same directory is what installs locally and what uploads:

```
<package>/
  mila.json              the manifest
  <weights>.safetensors  the artifact
  <tokenizer>.bin
  LICENSE                the source model's license text
  README.md              model card, including the statement that changes were made
```

`Distribution.ModelPackage` is that directory: `buildPackage` assembles it and derives every digest
from the bytes, and `validate` reads each declared file back and reports whether the package agrees
with its own manifest. A repository manifest covers every variant it publishes, so packaging FP8 into
a directory that already holds FP4 merges into the manifest rather than replacing it.

`Tools/ExportArtifact` drives it: `--package <dir>` after an export, `--validate <dir>` on its own,
and `--install <dir>` to publish to the local store.

Two grades of finding, because they have different consequences. A **problem** means the bytes
disagree with the manifest, and nothing may be emitted or installed. A **warning** means the package
would ship without its LICENSE or its model card, which is a publishing decision rather than a
corruption. A declared path that would leave the package directory is a problem: a manifest can
arrive from a hub, so the path in one is untrusted input.

**Publishing to the local store** installs the package: hash each file, move it into `blobs/`, write
the record. Move rather than copy -- it is free on one volume, and it keeps a single integrity model
in which the path is the digest, with no second class of file that a manifest merely points at.
`ModelStore::adoptBlob` is `ensureBlob`'s counterpart for bytes that need no transfer, and it hashes
rather than trusting the caller: a blob adopted unverified would poison every later cache hit. It
takes the same per-digest transfer lock, and across volumes -- where there is no atomic move -- the
bytes go through `tmp/` so a partial copy never occupies a path that implies verification.

Installing does not validate first. Adoption hashes each file as it takes it, so a separate
validation pass would read every byte a second time; at 6.8 GB that is not a cost worth paying for a
check that already happened.

**Publishing to a hub** validates the package and hands it to external tooling.
`Tools/Publishing/publish_model.py` does the upload through `huggingface_hub`: it validates digests
before uploading, skips files the hub already holds, and verifies afterward. The library contributes
the package and the validation; it does not contain an HTTP method that writes. It takes a package
directory and `--repo <owner>/<name>`; the older card directory, whose `publish.json` maps hub paths
onto large files kept outside it, still works and is what a package makes unnecessary.

The division is deliberate. Uploading to HuggingFace means the preupload check, the LFS batch API,
multipart transfer and a commit call -- a large failure surface, for a workflow a maintainer runs by
hand a few times per release, in a language that already has a maintained client.

---

## The management surface

```
ModelStore                     filesystem only, always available
  list()                    -> [StoredModel]     every installed record
  locate(coordinate)        -> StoredModel?      paths, or nothing
  describe(coordinate)      -> Manifest?
  remove(coordinate)        -> RemovalReport     record, then sweep
  prune()                   -> RemovalReport     unreferenced blobs, rejects, stale partials
  diskUsage()               -> StoreUsage        by model and in total
  install(package, options) -> StoredModel       verify, adopt, record

ModelPackage                   filesystem only, always available
  open(directory)           -> ModelPackage      reads the manifest, hashes nothing
  validate(variant)         -> PackageValidation problems and warnings
  buildPackage(request)     -> ModelPackage      assemble, merging into what is there

ModelHub                       network, gated
  listModels(owner)         -> [HubModel]
  describe(coordinate)      -> Manifest
  pull(coordinate, store, progress) -> StoredModel
```

Progress is a callback taking bytes-so-far and total, and the library does not rate-limit it. Deciding
how often to redraw is the consumer's problem, because a console line, a TUI and a server log want
three different answers.

---

## Consumers

**Chat.** The catalog stops being a table of file paths and becomes a table of aliases over
coordinates. Quantization stops being part of the alias and becomes the variant, which the coordinate
grammar already expresses:

| Before | After |
|---|---|
| `gemma-12b`, `gemma-12b-packed`, `gemma-12b-hub` | `gemma-12b` -> `mila-llm/gemma-4-12b-it`, variants `fp4`, `fp8` |
| `llama-3b`, `llama-3b-fp32` | `llama-3.2-3b-instruct-bf16`, `llama-3.2-3b-instruct-fp32` |

`/model <alias|coordinate> [variant]` selects; `/models` lists what is installed and what the hub
offers; `/pull` and `/rm` manage. Three aliases naming one model, distinguished by a provenance nobody
outside the codebase can decode, go away.

**The inference server.** Consumes the same store through the Python binding, in a separate process
from Chat. It lists and loads; it never pulls in response to a request.

**The binding.** `ModelStore` must be reachable from Python even in a build without the hub -- the
manylinux wheel cannot link libcurl, and listing or removing an installed model is not a network
operation. See [Build gating](#build-gating).

---

## Retiring `.bin`

The flat MILA container stops being a form Mila distributes or catalogues. Every catalogued model is a
safetensors artifact with a manifest.

**The reader keeps its MILA branch.** `Serialization.PretrainedReader` sniffs the leading magic and
fills the same tensor index from either container, so everything past the header parse is already
common. Removing that branch would buy nothing and would strand every `.bin` already on disk;
retiring the *format* is a catalogue and publishing decision, not a loader change.

Migration per model: export to safetensors, package, publish to the local store. `ExportArtifact`
already performs the export, including the BF16 passthrough case where no quantization is applied.
Models Mila may republish go to `mila-llm`; the rest stay under `local` on the machine that converted
them.

---

## Build gating

The split is by dependency, not by theme, and **the only optional thing is the transport.**

- **Always compiled** -- `Sha256`, `Environment`, `ModelCoordinate`, `ModelManifest`,
  `ModelPackage`, `ModelStore`, `ModelHub`, `ModelResolver`, `HttpTransport` and
  **`HuggingFaceHub`**: naming, the schema, layout, records, list, locate, remove, prune,
  package, validate, install, `pull` itself, and every HuggingFace URL shape, token rule,
  listing quirk and status meaning. None of it performs I/O. `HuggingFaceHub` holds an
  `IHttpTransport` and asks it for bytes.
- **`Distribution.HttpTransportBackend`** -- the transport, with two candidate source files and
  exactly one compiled. `HttpTransportBackend.Curl.ixx` brings `CurlHttpTransport` and keeps
  libcurl private; `HttpTransportBackend.Null.ixx` supplies one that refuses by name. Both
  export `makeDefaultHttpTransport()` and `kHttpTransportAvailable`.

**Why the seam is the transport and not the hub.** Gating `HuggingFaceHub` was tried and was
wrong: URL construction, token discovery, the `gated: "auto"` quirk and the 401-vs-403 distinction
are *knowledge*, and a host language that had to supply them would be reimplementing the library's
policy rather than lending it a capability. With the seam one level down, a caller supplies
`GET url -> bytes` and reimplements nothing.

`MILA_ENABLE_LIBCURL` defaults ON, matching `MILA_ENABLE_CUDA` and
`MILA_ENABLE_PYTHON_BINDINGS`.

**The reason is TLS, not libcurl.** The manylinux policy constrains dynamic linking only, and
Mila vendors libcurl statically -- `auditwheel` never sees it. What it does see is the system
OpenSSL that curl links on Linux, which is not whitelisted. The alternatives are to vendor
OpenSSL and own its CVEs, or to link it statically and ship a CA bundle; both give up the system
trust store, which is the property that makes a corporate proxy with an injected root work
untouched. Python has already solved this, so the wheel builds with the backend OFF and supplies
the transport itself. Secondarily, a library whose entire third-party surface is two headers
should not force a network dependency on a consumer that only loads from disk.

**Selection is by source file, never by `#ifdef`.** A `PUBLIC` macro deciding what a module
exports makes the BMI depend on the preprocessor, so `import Mila;` would no longer name one
thing. Choosing the file instead keeps both alternatives compiled code and leaves the variation
where it belongs: in the build configuration. A consumer that must say which build it is asks
`Distribution::kHttpTransportAvailable`.

### The HTTP client

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

---

## Licensing per family

The publishing story is not uniform, and assuming it is would be the first mistake.

| Family | License | Republishable to `mila-llm` |
|---|---|---|
| Gemma 4 | Apache 2.0 | Yes, public and ungated |
| Gemma 3 and earlier | Gemma Terms of Use | Terms propagate; gate to match Google |
| Llama 3.1 / 3.2 | Llama Community License | Propagates; Meta gates the source |

Gemma 4's Apache 2.0 still requires the license text, attribution, and a statement that changes were
made -- quantization is a modification. That belongs in the package alongside the root NOTICE.md habit
already established.

A family that cannot be republished is not a hole in the catalogue. It is a locally published model: the user
converts it once, the store lists it exactly like a fetched one, and `publish` refuses the hub
destination with the reason rather than failing at a 403.

---

## Decisions closed

1. **Revision pinning.** A coordinate accepts `@<revision>` and a record stores the resolved commit.
   One installed revision per variant; a pull at a different revision replaces the record and the old
   blobs become sweepable. Side-by-side revisions are not supported, because the disk cost is
   invisible at the point where a user would incur it.
2. **Manifest caching.** Records are persisted, so `list`, `describe` and `locate` are offline
   operations and startup never depends on the network. A `pull` always revalidates against the hub,
   since fetching a manifest is one small GET and a stale one silently pins a superseded artifact.
3. **Progress reporting.** Bytes-so-far and total, unthrottled by the library. Chat renders a bar; the
   current consumer-side gate fires on every chunk whose running percentage happens to be a multiple
   of five, which at 6.33 GB is hundreds of redraws per step rather than one.

---

## Build plan

Phases 1 to 5 landed in `0.20.0-beta.2+21..+25`: the HTTP client, the content-addressed cache, the
coordinate resolver, the Chat catalog entry, and the published `mila-llm/gemma-4-12b-it` repository.
What follows completes distribution as a managed system.

**Phase 6 -- the store.** Split `ModelStore` out from behind the hub option; add the record tree,
and write a record on every successful pull.
*Done when:* a pulled model appears in `list()`, and a build with no HTTP transport still lists
and locates it.

**Phase 7 -- management.** `remove`, `prune`, `diskUsage`, refcounted sweep, transfer lock.
*Done when:* removing one of two variants sharing a tokenizer leaves the tokenizer blob in place;
prune reclaims a `.rejected` file and a stale partial; two processes pulling one blob do not corrupt
each other.

**Phase 8 -- the hub interface.** `IModelHub` with `HuggingFaceHub` behind it, plus `listModels`.
*Done when:* the resolver names no HuggingFace URL, and listing `mila-llm` reports the published
models.

**Phase 9 -- packaging and publish.** Package assembly and validation; install to the local store;
hub-ready output handed to `publish_model.py`.
*Done when:* a converted model becomes an installed model that `list` reports and Chat loads, and the
same package validates for hub upload.

**Phase 10 -- the load boundary and the catalogue.** `locate` never touches the network; Chat's
catalog becomes aliases over coordinates; `.bin` leaves the catalogue.
*Done when:* a clean machine pulls and runs Gemma 4 through named commands, and no catalogue entry
names a `.bin`.

---

## Open items

- **The cold download has one unexplained failure.** A 6.33 GB Chat transfer failed its digest check
  (`expected d49c6c16..., got 8fe5cf53...`) while the same client fetched the exact digest at both
  14 MB and 6.33 GB through `ExportArtifact --fetch`. The leading explanation is a corrupt transfer
  that the integrity check caught -- the design working -- but it is unproven. A mismatch now reports
  the byte count and keeps the file as `.rejected`: exactly 6799927760 bytes with a wrong digest means
  altered in flight, any other count means a length bug.
- **Parallel range downloads.** A single connection may not saturate the link. Measure before
  building; the CDN's single-connection throughput is the suspect, not the client.
- **`publish_model.py` hashes each large file twice**, once to validate and once to decide whether the
  hub already holds it.
