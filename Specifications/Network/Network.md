# Networking

Status: **proposed**, 2026-11-15 (Todd asked for it). **Phase 1, the move, is built** (Todd directed it the same day,
section 7); phases 2 to 4 are still proposals. The layer the
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

- **One User-Agent**, `Mila/<version>` from `Mila.Version`, used by every transport that adds one. Today there are
  three copies of `"Mila/0.20"` (curl, URLSession, Network.framework; `Issues/Untriaged.md`).

## 5. The client serves every use

`HttpClient` was shaped by the store: GET, a token, resume by `Range`, a sink for a large body. The page reader and
web search need four things it lacks, which is why they went round it:

| Need | Who | Proposed |
|---|---|---|
| A request body (POST) | web search | `HttpRequest::body`; the transport contract already carries one (`HttpClient.md` 4.1) |
| A bounded body that reports the cut instead of failing | page reader (4 MB), web search (1 MB) | `HttpRequest::maximum_body_bytes`; `HttpResult` says whether it was cut |
| A header that must not cross a change of host | web search's `Authorization` | today's `token` field (always sent as `Bearer`), generalised: a header the request marks as host-bound is dropped when a redirect changes host |
| Redirects that must stay web addresses | page reader | `http` and `https` only, for every user. The store's redirects already are |

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
3. **The client serves every use** (section 5). The page reader and web search go through `HttpClient`.
   *Done when:* neither calls `fetch` itself; a test drops a host-bound header across a change of host; the page
   reader's tests and `ToolGrant.Network.cpp` pass unchanged.
4. **The build option's name**, if section 8 decides it.

## 8. Open

- **`MILA_ENABLE_LIBCURL`** names the wrong thing on a Mac (`HttpClient.md` 8). With networking a layer, the option
  says whether the library reaches the network at all, and the platform picks the transport:
  `MILA_ENABLE_NETWORKING`, after the layer, is the obvious name. The rename reaches CI, the Dockerfiles, `getting-started.md` and
  user-visible strings, so it is phase 4, not phase 1.
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
