# HTTP Client

Status: **proposed**, 2026-08-05. Supersedes the transport half of the Build gating section in
[ModelDistribution.md](ModelDistribution.md). The layer the client and its transports belong to, `Mila::Networking`, and
the move out of `Distribution` are [Network/Network.md](Network/Network.md); the move is built (phase 1, 2026-11-15),
so the module names below are `Networking.*` and the namespace `Mila::Networking`.

---

## 1. What went wrong

libcurl was chosen so that one HTTP implementation would serve every platform: write once, and
Windows and Linux both just work. That is not what we got, and the reason is worth stating exactly,
because it is not an oversight that can be patched.

libcurl unified the **client**. It never unified **TLS trust**. `CURL_USE_SCHANNEL ${WIN32}` gives
Schannel on Windows and the system OpenSSL on Linux -- deliberately, so that there is no CA bundle
to ship or keep current and a corporate proxy with an injected root works untouched. One API over
two stacks.

It is the TLS divergence, not the client, that broke packaging: OpenSSL is not on the manylinux
whitelist, so the Linux wheel cannot link the thing the whole design rests on.

**There is no third door here.** Every HTTP stack faces the same fork -- use the operating system's
trust store and accept platform-specific linkage, or bundle your own and accept owning it. We chose
the OS store. That choice was right and it is why "write once" could never have held all the way
down.

## 2. What is actually correct

A host-supplied transport is not a workaround for manylinux. It is the correct behaviour, and
manylinux is merely what forced us to notice.

A C extension that does its own networking inside a Python process ignores `HTTPS_PROXY`, ignores
`REQUESTS_CA_BUNDLE`, ignores the corporate CA the user already configured for pip, and ignores
their `~/.netrc`. It is a badly-behaved guest: quietly using different network policy from
everything else in that process. **In a host with its own network stack, the host's stack is the
right one.**

So the rule is: one client for native builds, the host's transport when Mila is a guest. That is
not a compromise, and it is not what is wrong today.

## 3. What is wrong today

The redirect hop limit, the authorization-header drop on a change of host, the `Range` request, the
200-versus-206 rule and the status mapping all live **inside** the libcurl implementation.

Consequences, in order of severity:

1. **The security-relevant rule is per-transport.** The token drop across a host change exists once
   per implementation, so every new transport is a fresh chance to leak a bearer token to a CDN.
   The Python side happens to be safe because `requests` strips `Authorization` across hosts on its
   own -- that is luck we are relying on, not a contract we enforce.
2. **A second platform client is a duplication event.** WinHTTP or CFNetwork would each reimplement
   all of it, which is the same mistake one level down that moving the seam to the transport just
   corrected one level up.
3. **The policy is not testable without a transport.** `HttpClient.Cpu.cpp` today tests
   `resolveRedirect`, a pure string function, because everything else is welded to curl. The hop
   limit, the token drop and the Range rule -- the parts worth pinning -- have no coverage at all.

## 4. The design

Three layers, each with one job.

```
  HttpClient          policy: redirects, token drop, Range/resume, status mapping
        |             always compiled, depends on nothing
        v
  IHttpTransport      one request, no interpretation
        |
        +-- CurlHttpTransport      libcurl            (gated: MILA_ENABLE_NETWORKING; Windows, Linux)
        +-- UrlSessionHttpTransport  Apple's URL loading  (macOS, section 9)
        +-- NetworkHttpTransport   Network.framework  (macOS, a grant's network, section 9)
        +-- WinHttpTransport       WinHTTP            (future, no dependency)
        +-- DelegateHttpTransport  a host callable    (the Python wheel)
```

This also settles the naming question. Today's file is both layers welded together, which is why it
answered to neither name. Split, each takes its right one: **a transport moves bytes for one
request; a client decides how to ask and how to read the answer.**

### 4.1 IHttpTransport -- the narrow contract

```cpp
export struct HttpHeader { std::string name; std::string value; };

export struct HttpFetch
{
    std::string url;
    std::vector<HttpHeader> headers;      // exactly what to send; nothing is added
    long low_speed_timeout_seconds{ 60 };
    std::optional<std::string> body;      // a POST of these bytes when set, a GET when not
};

export struct HttpResponse
{
    long http_code{ 0 };                  // 0 when the request never completed
    std::string location;                 // Location header, empty when absent
    uint64_t content_length{ 0 };
    bool transport_failed{ false };       // connection, TLS or I/O -- not an HTTP status
    std::string message;                  // detail on failure; never contains a token
};

export class IHttpTransport
{
public:
    virtual ~IHttpTransport() = default;
    virtual std::string name() const = 0;
    virtual HttpResponse fetch( const HttpFetch& request, const SinkCallback& sink ) const = 0;
};
```

Three obligations, and they are the whole contract:

- **Do not follow redirects.** Report the status and the `Location` header; the client decides.
  (`CURLOPT_FOLLOWLOCATION` is already off; WinHTTP wants `WINHTTP_OPTION_DISABLE_REDIRECTS`;
  `requests` wants `allow_redirects=False`.)
- **Send exactly the headers given.** No token discovery, no `Range` construction, no additions.
- **Do not deliver a non-2xx body to the sink.** Capture a bounded prefix into `message` instead.
  A redirect or error body must never reach a caller that is hashing bytes into a blob.

Nothing security-relevant is left to a transport. It cannot leak a token because it is never told
one -- it is handed headers that the client has already decided are safe for that exact host.

**A body makes the request a POST** (2026-10-26, for Mila Assistant's web search, whose provider takes only POST). The
model store never sends one, and `HttpClient` has no body to pass; a caller with one goes to the transport directly. A
transport that cannot send a body -- the Python binding's delegate, whose signature carries none -- answers such a
request as a transport failure rather than sending it as a GET.

### 4.2 HttpClient -- the policy

Concrete, always compiled, no dependency. Owns everything the transports used to each own:

- Follows redirects to `maximum_redirects`, resolving relative `Location` values (all four forms;
  HuggingFace answers a manifest request with a 307 to a root-relative path), and reports
  `TooManyRedirects` past the bound.
- **Fetches only `http` and `https` addresses**, the first and every redirect (`isWebAddress`); a
  redirect elsewhere, or one with no `Location`, is `RedirectRefused`.
- **Drops the `Authorization` header when the host changes.** One place. This is the rule that
  matters, and it is now impossible for a transport to get wrong. The request's own `headers` go to
  every host it reaches, so one naming `Authorization` is refused (`std::invalid_argument`): a secret
  is only ever the `token`.
- **Sends a body as a POST** (`HttpRequest::body`), and follows no redirect for it:
  `RedirectRefused`.
- **Bounds the body** at `maximum_body_bytes`: the sink is given the bytes up to the bound, the
  transfer stops, and the result says `cut`. A body of exactly the bound is not cut.
- Builds `Range: bytes=N-` for a resume, and reports `RangeIgnored` when the answer is 200 rather
  than 206 -- the server is sending the whole file, and appending that to a partial silently
  concatenates.
- Maps status to `HttpStatus`, keeping 401 and 403 distinct because one means "get a token" and the
  other means "accept the terms". What a status means to a person is the caller's to say, from the
  status or `http_code`.
- `fetch` streams to a sink; `fetchString` buffers, for manifests, pages and API replies. Every
  request in the library and the applications goes through one or the other (`Network/Network.md`
  section 5).

Its public surface is today's `HttpRequest` / `HttpResult` / `HttpStatus`, unchanged, so
`HuggingFaceHub` is untouched by this work.

### 4.3 What each transport becomes

| transport | size | notes |
|---|---|---|
| `CurlHttpTransport` | small | today's `httpGet` minus the redirect loop, the token logic and the Range handling |
| `WinHttpTransport` | small | becomes a considered option rather than a duplication event; no dependency, Schannel already underneath |
| `DelegateHttpTransport` | trivial | the host returns `(http_code, location, content_length, message)` and streams the body |

The Python side of the wheel shrinks to roughly this, and notably never sees a token it should not
send, never decides a redirect, and never interprets a status:

```python
def transport(url, headers, sink):
    with requests.get(url, headers=headers, stream=True,
                      allow_redirects=False) as response:
        if 200 <= response.status_code < 300:
            for chunk in response.iter_content(1 << 20):
                if not sink(chunk):
                    break
        return HttpResponse(status_code=response.status_code,
                            location=response.headers.get("Location", ""),
                            content_length=int(response.headers.get("Content-Length", 0)))
```

## 5. What this buys

- **The token rule exists once**, above every transport, in code with no dependency.
- **A platform client stops being expensive.** Whether WinHTTP ever happens becomes a cost question
  about one small class, not a question about reimplementing policy.
- **The policy becomes testable offline**, in the always-compiled test set, against a fake transport
  that returns scripted responses: the hop limit, a redirect chain, a cross-host hop that must drop
  the header, a 200 answering a `Range`. None of that has coverage today.
- **`MILA_ENABLE_NETWORKING` (once `MILA_ENABLE_LIBCURL`) stops being interesting.** It selects which raw fetcher compiles. No
  behaviour rides on it, and the null case is a transport that reports `transport_failed` rather
  than a hole in the API.

## 6. What this does not do

It does not restore literal write-once. That is available only by bundling TLS with curl -- static
OpenSSL or mbedTLS plus a shipped CA bundle -- which makes one stack serve every platform including
wheels, at the price of owning a TLS stack's CVE response and giving up the system trust store.
**Not recommended** for a project this size, and it would discard the corporate-proxy property the
current build deliberately has.

## 7. Build plan

1. **`Networking.HttpTransport`** (always) -- `IHttpTransport` and `SinkCallback`, with `HttpHeader`, `HttpFetch` and
   `HttpResponse` in modules of their own.
2. **`Networking.HttpClient`** (always) -- `ProgressCallback`, `HttpClient`, and `resolveRedirect` / `hostOf` moved up
   from the curl file, with `HttpStatus`, `HttpRequest` and `HttpResult` in modules of their own.
3. **`Networking.CurlHttpTransport`** (gated) -- reduced to one `fetch`.
4. **`Networking.HttpTransportBackend`** -- `.Curl` / `.Apple` / `.Null`, `makeDefaultHttpTransport()`,
   `kHttpTransportAvailable`.
5. **`HuggingFaceHub`** takes an `HttpClient` rather than a transport -- a one-line change, since
   the client's surface is what it already calls.
6. **Tests** -- a `ScriptedTransport` in the always-compiled set covering the hop limit, the
   cross-host token drop, relative `Location` resolution, and `Range` answered 200.
7. **Binding** -- `HttpGetDelegate` becomes the transport shape above.

*Done when:* the cross-host token drop has a test that fails if the rule is removed, and both
presets are green.

## 8. Open

- **Bounded error-body capture.** Set at **4096 bytes** for every transport
  (`kMaximumCapturedErrorBytes`, `HttpTransport.ixx`): enough for an error page's useful line, bounded enough that a
  hostile server cannot grow it. Revisit if a real message is ever truncated.
- **A delegate transport reports its headers late.** `IHttpTransport` gets a `HeadersCallback`
  fired before the body, which `CurlHttpTransport` drives from libcurl's header hook. A delegate
  cannot: the host returns everything at once, so `DelegateHttpTransport` fires it after the
  transfer. The consequence is that a progress callback in a hosted build sees `total == 0` while
  bytes arrive. Fixable by letting the host announce the response through the sink object before
  streaming; not worth the extra call shape until something needs the total.
- **Proxy configuration** is currently whatever libcurl picks up from the environment. Under a
  host transport it is whatever the host does. That difference is intended -- see section 2 -- but
  it is undocumented behaviour today. On macOS it is the Mac's own proxy settings (section 9).
- **`MILA_ENABLE_LIBCURL` named the wrong thing on a Mac**, where ON selects Apple's transports and
  no libcurl is built (section 9). It is `MILA_ENABLE_NETWORKING` now (`Network/Network.md` 7, phase 4).

## 9. macOS: the operating system's transports

*Decided 2026-11-14 (Todd).* Section 1's fork has one answer on a Mac: Apple's own stack. macOS has no
OpenSSL an application may link, curl removed its Secure Transport backend in 2025, and Homebrew's
OpenSSL trusts Homebrew's certificate file rather than the Keychain, so a Mac build that linked it ran
only where Homebrew is installed. curl is not built on macOS; bundling a TLS stack is section 6's
rejected door, and stays rejected (Todd: "2 is not an option").

Two transports, because a grant's network needs what Apple's URL loading does not offer:

- **`UrlSessionHttpTransport`** is `makeDefaultHttpTransport()`: the model store's listings and
  downloads. Keychain trust, the Mac's proxy settings (System Settings, automatic configuration
  included) and Apple's TLS, through an ephemeral session with no cache and no cookies. A redirect is
  refused in the session's delegate and reported, as the contract requires. Headers are set as given;
  Apple lists `Authorization` among the headers its loader manages, and its developer support says
  setting it directly is the only way to send a bearer token, which is what is done. Two are added when
  the client sends none, so the bytes are what curl's request brings: Mila's `User-Agent`
  (`userAgent()`, `Network/Network.md` 4.3), and
  `Accept-Encoding: identity`, since the loader otherwise asks for compression and decodes it, and a
  resumed or hashed download must see the resource's own bytes. Apple's code is Objective-C, reached
  through one `.m` file behind a C interface (`Networking/Transports/Apple/Implementation/UrlSessionFetch.m`), with no C++ in it.
- **`NetworkHttpTransport`** is `makeCheckedHttpTransport()`: a tool grant's network
  (`ToolCalling.md` 3.4). Its guarantee is curl's: the check decides on the address connected to, after
  the name was resolved, before any packet reaches it, and no proxy is used. URL loading resolves and
  connects inside itself and reports the address only once the request has been sent, so it cannot
  keep that. Network.framework, a C API, can: the transport resolves the name itself, puts each address
  to the check, and connects to that address, never the name, with TLS verified against the URL's host
  name and the system's trust. Before a byte is sent it confirms from the connection's own path that
  data goes to that address, and from its establishment report that no proxy was used --
  `nw_parameters_set_prefer_no_proxy` is a preference, and a proxy can still be taken. Network.framework
  moves bytes and no more, so this transport writes the request and reads the reply itself: HTTP/1.1,
  one request per connection (`Connection: close`), only `http` and `https`, through
  `HttpReplyReader`. The connection code is C (`Networking/Transports/Apple/Implementation/NetworkConnection.c`).
- **`HttpReplyReader`** reads an HTTP/1.1 reply as its bytes arrive: the status line and headers,
  bounded, then a body by `Content-Length`, by chunks, or to the close; an interim 1xx reply is
  passed over. It is plain C++ in every build, so its rules are tested on every platform.

The grant's network tests (`ToolGrant.Network.cpp`) run on every platform, their loopback server on
each platform's sockets. The default transport's use of a proxy named in the environment is curl's
behaviour and is tested where curl is built; a Mac's default transport takes the Mac's proxy settings.
