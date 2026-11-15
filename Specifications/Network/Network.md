# Networking

Status: **proposed**, 2026-11-15 (Todd asked for it). **Phase 1, the move, is built** (Todd directed it the same day,
section 7), and so are **phase 2, one User-Agent, phase 3, the client serving every use, and phase 4, the build
option's name** (each agreed with Todd the same day). The layer the
library reaches the network through, its place among the others, what it holds and what it does not, and the work
that moves today's code into it. The HTTP client's own design stays in [HttpClient.md](../HttpClient.md); this
document decides the layer around it.

---

## 1. The problem

The HTTP client was written for the model store, so it was put under `Distribution/Http/` and named
`Mila::Distribution`. It has since gained three users that have nothing to do with distributing a model:

- **The web page tool** (`Agents/ProvidedTools/WebPageReader.ixx`) reads pages through a tool grant's transport.
- **The grant itself** (`Agents/Grants/ToolGrant.ixx`) makes the checked transport (`makeCheckedHttpTransport`),
  and the address classification it checks with (`Agents/Grants/PublicAddress.ixx`) lives in `Agents`.
- **Mila Assistant's web search** (`Apps/Desktop/Session/Src/Tools/Desktop.WebSearch.ixx`) posts to a search
  service through the same transport.

So `Agents` imports `Distribution` to reach the network, and an application names `Mila::Distribution::HttpFetch`
to call a search API. Coming next are the MCP transport over HTTP (`ToolCalling.md` 6) and a hosted model reached
as an agent's resource. Neither has anything to do with distribution either.

The placement hides a second problem. `HttpClient.md` 5 promised that HTTP policy exists once, above every
transport. That holds for the model store, but both users outside it go to the transport directly and keep their
own policy:

- `WebPageReader::fetch` follows redirects in its own loop, calls `resolveRedirect` itself, and maps statuses
  itself.
- `WebSearch::search` sends a bearer token in a header it builds and maps statuses itself.

Neither is wrong today. The page reader sends no token, and web search follows no redirect. But each is the
duplication `HttpClient.md` 3 named, one layer up.

## 2. Decisions proposed

1. **`Mila::Networking` is a folder of the Mila library, not a library of its own.** `Lib/Src/Networking/`, in the
   `Mila` target like `Dnn`, `Distribution`, `Deployment` and `Agents`, brought by `import Mila;`. One library,
   one install (`Direction.md` decision 2, 2026-11-02). Its edges are held by `Publishing/hooks/check_layers.py`,
   as every folder's are.
2. **It is the lowest layer that does I/O to another machine, and imports only `Utils` and the root**
   (`Mila.Version`, for the User-Agent). Nothing in it knows of a model, a tool, an agent or an application.
3. **It is an outbound client.** HTTP(S) requests, the transports that carry them, and the address checks a
   connection can be held to. It is not a server and not a sockets library. MIS's server is Python, and an MCP
   server Mila talks to is reached as a client.
4. **Platform transports are selected by the build, not by a type.** Compute backends are a compile-time axis
   (`OperationTraits` keyed on `DeviceType`), because one program can run on several devices and generic code is
   written over them. A transport is not: the platform fixes it, and on a Mac its purpose does (plain or checked,
   `HttpClient.md` 9). No code is written generic over transports. The model is the grant's file system split
   (`Agents.GrantFileSystem`, `.Windows` / `.Posix`): one module name, one source file chosen by CMake. The folder
   layout still mirrors `Dnn/Compute/Devices/`, one folder per backend.
5. **macOS and Windows are peers** (Todd, 2026-11-15: macOS is a platform with parity to Windows). Every rule of the
   transport contract and of the checked connection holds on both, and is tested on both: a test that runs on one
   and not the other is a gap, not a platform difference. A transport may differ only where `HttpClient.md` says
   the platform's own policy is the right one, such as the Mac's proxy settings.
6. **The client is the only way to make a request.** Every user in the library and the applications goes through
   `HttpClient`, so redirects, the token drop on a change of host, body bounds and status mapping exist once
   (section 5). A transport stays public, because the Python binding and tests implement it, but nothing calls one
   directly.

## 3. Where it sits

```
Applications        Mila Assistant   Chat   MIS   mila-store   a developer's program
                                 \     |     |      |      /
Mila::Agents        Agent, Session, ToolGate, grants, provided tools
                       |                  |                         |
Mila::Deployment    planDeployment, load  |                         |
                    /            \        |                         |
Mila::Dnn, Data   models ...    Mila::Distribution  hub, store, packaging
                                          |                         |
                                          +------------+------------+
                                                       |
Mila::Networking       HttpClient, IHttpTransport, HttpReplyReader, address classification
                                                       |
  transports        Curl (Windows, Linux) | Apple (URLSession, Network.framework) | Null | host delegate (binding)
                                                       |
Utils, root (Mila.Version)
```

It's a graph, not a stack. `Agents` imports `Networking` directly (the grant's transport, the page reader, a future
MCP client) and `Distribution` too (the store). `Deployment` imports `Distribution` and never `Networking`; `Dnn`
imports neither. `check_layers.py` becomes:

| Folder | May import |
|---|---|
| `Networking` | `Utils`, (root) |
| `Distribution` | `Networking`, `Utils`, (root) |
| `Agents` | as today, and `Networking` |

## 4. What it holds

### 4.1 Moved from `Distribution/Http/`

| Today | In `Networking/` |
|---|---|
| `HttpTransport.ixx`: `HttpHeader`, `HttpFetch`, `HttpResponse`, `SinkCallback`, `HeadersCallback`, `ConnectionCheck`, `IHttpTransport`, `kMaximumCapturedErrorBytes` | Split by the one-type-per-file rule, since the move touches it: `HttpHeader.ixx`, `HttpFetch.ixx`, `HttpResponse.ixx`, `HttpTransport.ixx` (the interface, with its callback aliases and the capture bound beside it) |
| `HttpClient.ixx`: `HttpStatus`, `HttpRequest`, `HttpResult`, `HttpClient`, `hostOf`, `resolveRedirect` | `HttpStatus.ixx`, `HttpRequest.ixx`, `HttpResult.ixx`, `HttpClient.ixx` (with `hostOf` and `resolveRedirect` beside it, or in `Url.ixx` if section 4.2's URL reading joins them) |
| `HttpReplyReader.ixx` + `Implementation/HttpReplyReader.cpp` | As they are |
| `HttpTransportBackend.{Curl,Apple,Null}.ixx` | `Transports/HttpTransportBackend.{Curl,Apple,Null}.ixx`, still one module name |
| `CurlHttpTransport.ixx` | `Transports/Curl/` |
| `UrlSessionHttpTransport`, `NetworkHttpTransport`, `UrlSessionFetch.{h,m}`, `NetworkConnection.{h,c}` | `Transports/Apple/` |

### 4.2 Moved from elsewhere

- **Address classification**, `isPublicAddress` and its table (`Agents/Grants/PublicAddress.ixx`). Whether an
  address is globally reachable is a fact about the address, and IANA's registries decide it. Applying it to a
  tool's network is the grant's policy, and that stays in `Agents` (`ToolCalling.md` 3.4). `ToolGrant` keeps the
  call `makeCheckedHttpTransport( isPublicAddress )`; only the function's home moves.
- **Reading a URL.** `NetworkHttpTransport`'s `readUrl` (scheme, host, IPv6 brackets, port, path) and
  `WebPageReader::isWebAddress` answer the same question twice. If section 5 is taken, one reading serves both.

### 4.3 New

- **One User-Agent**, `Mila/<version>` from `Mila.Version`, used by every transport that adds one. The version is
  the full one, dev tail included (`Mila/0.21.0-dev+145`; `Mila/0.21.0` at a tag), so a server's log tells a
  development build from a release (Todd, 2026-11-15). Before phase 2 there were three copies of `"Mila/0.20"`
  (curl, URLSession, Network.framework).

## 5. The client serves every use

`HttpClient` was shaped by the store: GET, a token, resume by `Range`, a sink for a large body. The page reader and
web search need four things it lacks, which is why they went round it:

| Need | Who | Decided |
|---|---|---|
| A request body (POST) | web search | `HttpRequest::body`; the transport contract already carries one (`HttpClient.md` 4.1). A request with a body follows no redirect (`RedirectRefused`): no user needs one, and a redirected POST has rules of its own. `get` and `getString` became `fetch` and `fetchString`, since a `get` that posts misleads (Todd, 2026-11-15) |
| Headers of the request's own | page reader's `Accept`, web search's `Content-Type` | `HttpRequest::headers`, sent on every hop; one naming `Authorization` is refused, so a secret can only be the token |
| A bounded body that reports the cut instead of failing | page reader (4 MB), web search (1 MB) | `HttpRequest::maximum_body_bytes`; `HttpResult::cut`. The page reader reads a cut page; web search refuses a cut answer, which before was parsed as broken JSON |
| A header that must not cross a change of host | web search's `Authorization` | today's `token` field, unchanged: web search's key is a `Bearer` token, which the field already carries and drops on a change of host. Generalising it to any host-bound header waits for a user whose secret is not a `Bearer` token (Todd, 2026-11-15) |
| Redirects that must stay web addresses | page reader | `http` and `https` only, for every user, the first address and each redirect (`isWebAddress`, beside the client, and the page reader's own check uses it). The store's redirects already are |

The status-to-message mapping stays with each caller, since the words are its own ("the site refused to show this
page", "the search service refused the key"). The client reports the status; the caller says what it means to its
reader.

The grant's transport is the client's transport: `HttpClient( grant.transport() )`. Each hop of a redirect is a new
`fetch`, so the check still runs on every connection.

## 6. What it does not do

- **No tools layer.** Whether `Tool`, its schema and the provided tools become a folder below `Agents` is a
  separate question with its own open parts: where `ToolGate` and the grants go, and `ToolCalling.md` 5. It is not
  part of this change.
- **No change to `Distribution`'s contents.** Hub, store, packaging and `Environment` stay. `Environment` reads a
  variable, and nothing in `Networking` needs it.
- **No change to any transport's behaviour** in the move (phase 1). Section 5 changes behaviour and is its own phase.
- **No new transport.** WinHTTP stays `HttpClient.md` 4.3's option.

## 7. Phases

Each phase is its own change.

1. **The move.** `Lib/Src/Networking/` as in section 4.1 and the address classification in 4.2, namespace
   `Mila::Networking`, modules `Networking.*`, the one-type-per-file split, `Mila.ixx`, `Lib/CMakeLists.txt` (module
   lists, implementation units, the Apple block with its frameworks and the `.m` file's `-fobjc-arc`, the curl
   block), `check_layers.py`, and every user: `Distribution`'s hub, the grant, the page reader, the Python
   binding's delegate (`DelegateHttpTransport`), Chat's catalogue, Mila Assistant's session (its models and web
   search), `mila-store`, `Tools/ExportArtifact`. `kHttpTransportAvailable` becomes `Mila::Networking`'s. Nothing
   changes behaviour: every transport, test and message does what it did. Tests move to `Lib/Tests/Networking/`
   (`HttpClient.Cpu.cpp`, `HttpReplyReader.Cpu.cpp`, `PublicAddress.cpp`); `ToolGrant.Network.cpp` stays with the
   grant, as do `WebPageReader.cpp` and `ModelHub.Cpu.cpp`. The specs that name the old modules (`HttpClient.md`,
   `ModelDistribution.md`, `ToolCalling.md`, `MilaDesktop.md`, `Metal/MetalBackend.md`, the index row in
   `Specifications/README.md`) are updated in the same change, and this section records phase 1 as built.
   *Done when:* every target builds on the PC and the Mac, the full suite passes on both, `check_layers.py`
   passes with section 3's table, and a search finds no `Distribution.Http`, `Mila::Distribution::Http`,
   `IHttpTransport` under `Distribution`, or `Agents.PublicAddress` left.
   *Built* (2026-11-15), as section 4.1's table has it, with these choices made on the way:
   - **The namespace is `Mila::Networking`, not `Mila::Network`** (Todd, 2026-11-15), and the folder, the modules
     and the tests' folder follow it. `Mila::Network` collided with `Mila::Dnn::Network`, the class template a model
     is built on: code inside a `Mila::` namespace other than `Dnn` with `using namespace Mila::Dnn;` found both for
     `Network<...>` (MSVC C2872), which the build met in `Lib/Tests/Dnn/Modeling/Network.Cuda.cpp` and the MNIST
     sample. `Mila::Net` was weighed and refused: an abbreviation, and "net" already names a neural network in
     the library's own docs and component types. This document keeps its path.
   - Each module whose signatures name a type re-exports its module (`Networking.HttpTransport` brings `HttpFetch` and
     `HttpResponse`, `Networking.HttpClient` brings the transport, `HttpRequest` and `HttpResult`), so a user that
     imported one module before imports one now.
   - `hostOf` and `resolveRedirect` stay beside the client (section 8's third point, for now). The client's bodies,
     and theirs, moved to `Networking/Implementation/HttpClient.cpp`, by `CLAUDE.md`'s rule for a plain class.
   - `NullHttpTransport` is no longer exported: only its own file names it.
   - `Agents.MxcProgramContainer` also imported the address module, for `nonPublicRanges()`; it imports
     `Networking.PublicAddress` now. `ContainedPrograms.md` 7 is updated with it.
   - `ToolCalling.md`, `MilaDesktop.md` and `Metal/MetalBackend.md` name no moved module, so nothing changed in
     them.

   *Done-when met* at `0.21.0-dev+144`: every target builds on the PC (`x64-claude-verify`) and the Mac
   (`macos-clang-release`); the full suites fail only where they did before the move (PC: the 4070's drafter near-tie
   and the three tests on the 26B stub, and the MXC tests until `MILA_MXC_LIBRARY` is set, then 21 of 21; Mac: the six
   Llama FP8 gates on Metal); `check_layers.py` passes; the search finds none of the old names.
2. **One User-Agent** (section 4.3). *Done when:* one definition, and a test reads it from each transport's
   request where the transport can be observed (curl and Network.framework, against the loopback server).
   *Built* (2026-11-15): `userAgent()` in `Networking.HttpTransport`, beside the capture bound, since the contract
   names it: a transport adds it only when the request names no User-Agent. curl and Network.framework call it;
   URLSession's Objective-C cannot import a module, so `MilaUrlSessionRequest` carries it as `user_agent`, filled
   by the C++ side. URLSession could be observed after all (it reaches `127.0.0.1` like the others), so
   `Lib/Tests/Networking/HttpTransport.UserAgent.cpp` reads the request every transport of the build sent, the
   plain and the checked: one User-Agent, the library's, and a request's own in its place when it names one. The
   loopback server moved to `Lib/Tests/Common/LoopbackServer.h` for it, shared with `ToolGrant.Network.cpp`.
3. **The client serves every use** (section 5). The page reader and web search go through `HttpClient`.
   *Done when:* neither calls `fetch` itself; a test drops a host-bound header across a change of host; the page
   reader's tests and `ToolGrant.Network.cpp` pass unchanged.
   *Built* (2026-11-15), with section 5's decisions. The host-bound header is the token, so the existing
   `DropsTheTokenWhenTheHostChanges` is that test, with a new one beside it that sends the request's own headers on
   every hop while the token stays with the first host; others cover the refused `Authorization` header, the body
   and its refused redirect, the cut, and the `http`/`https` rule. Two statuses were added so a caller can say
   which redirect failure it met without reading a message: `RedirectRefused` (no `Location`, not `http` or `https`,
   or a request with a body) and `TooManyRedirects`; both were `TransportError` before, and the two client tests that
   named it now name them. The page reader's tests pass unchanged; `ToolGrant.Network.cpp` changed only by the
   rename. The page reader's bodies moved to `Agents/ProvidedTools/Implementation/WebPageReader.cpp`, by `CLAUDE.md`'s
   rule for a plain class, since the change rewrote its fetch. Web search has no test, and nothing lets one give it
   a scripted server (`Issues/Untriaged.md`).
4. **The build option's name.** `MILA_ENABLE_LIBCURL` named the wrong thing on a Mac (`HttpClient.md` 8), where ON
   selects Apple's transports and no libcurl is built. With networking a layer, the option says whether the library
   reaches the network with a transport of its own, and the platform picks which: `MILA_ENABLE_NETWORKING`.
   *Built* (2026-11-15, Todd asked for it): the option, its uses in the three CMake files, the presets, CI, the
   Dockerfiles, `getting-started.md`, `NOTICE.md`, the binding's notes, and the user-visible strings in Chat,
   `mila-store` and the null transport. A build directory configured before carries the old name in its cache, and
   a consumer may still pass it, so the root `CMakeLists.txt` reads `MILA_ENABLE_LIBCURL` once into the new name with
   a warning and removes it from the cache. The preset `x64-debug-no-libcurl` keeps its name: it is a Windows
   preset, where OFF does mean no libcurl, and a renamed preset is a new build directory.

## 8. Open

- **Whether `HttpClient.md` folds into this document** once phase 3 lands, as `Network/HttpClient.md`. Moving a
  spec is its own commit (`Specifications/README.md` rule 4).
- **The one-type-per-file split of `HttpClient.ixx`'s helpers**: `hostOf` and `resolveRedirect` beside the client,
  or a `Url.ixx` with the URL reading of section 4.2. Phase 1 left them beside the client.

## 9. Relationship to other specs

- `HttpClient.md`: the client's policy and the transport contract. Unchanged in substance; its module names follow
  phase 1.
- `ToolCalling.md` 3.4: the grant's network rule. Unchanged; the classification it names moves home.
- `Direction.md` 3: the layering diagram gains `Mila::Networking` below `Mila::Distribution` when this is agreed.
- `ModelDistribution.md`: the store and hub reach the network through this layer.
