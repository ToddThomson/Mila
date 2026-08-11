# HTTP Client

Status: **proposed**, 2026-08-05. Supersedes the transport half of the Build gating section in
[ModelDistribution.md](ModelDistribution.md).

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
        +-- CurlHttpTransport      libcurl            (gated: MILA_ENABLE_LIBCURL)
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

### 4.2 HttpClient -- the policy

Concrete, always compiled, no dependency. Owns everything the transports used to each own:

- Follows redirects to `maximum_redirects`, resolving relative `Location` values (all four forms;
  HuggingFace answers a manifest request with a 307 to a root-relative path).
- **Drops the `Authorization` header when the host changes.** One place. This is the rule that
  matters, and it is now impossible for a transport to get wrong.
- Builds `Range: bytes=N-` for a resume, and reports `RangeIgnored` when the answer is 200 rather
  than 206 -- the server is sending the whole file, and appending that to a partial silently
  concatenates.
- Maps status to `HttpStatus`, keeping 401 and 403 distinct because one means "get a token" and the
  other means "accept the terms".
- Keeps `getString` for manifests and API responses.

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
- **`MILA_ENABLE_LIBCURL` stops being interesting.** It selects which raw fetcher compiles. No
  behaviour rides on it, and the null case is a transport that reports `transport_failed` rather
  than a hole in the API.

## 6. What this does not do

It does not restore literal write-once. That is available only by bundling TLS with curl -- static
OpenSSL or mbedTLS plus a shipped CA bundle -- which makes one stack serve every platform including
wheels, at the price of owning a TLS stack's CVE response and giving up the system trust store.
**Not recommended** for a project this size, and it would discard the corporate-proxy property the
current build deliberately has.

## 7. Build plan

1. **`Distribution.HttpTransport`** (always) -- `HttpHeader`, `HttpFetch`, `HttpResponse`,
   `SinkCallback`, `IHttpTransport`.
2. **`Distribution.HttpClient`** (always) -- `HttpStatus`, `HttpRequest`, `HttpResult`,
   `ProgressCallback`, `HttpClient`, and `resolveRedirect` / `hostOf` moved up from the curl file.
3. **`Distribution.CurlHttpTransport`** (gated) -- reduced to one `fetch`.
4. **`Distribution.HttpTransportBackend`** -- `.Curl` / `.Null`, `makeDefaultHttpTransport()`,
   `kHttpTransportAvailable`.
5. **`HuggingFaceHub`** takes an `HttpClient` rather than a transport -- a one-line change, since
   the client's surface is what it already calls.
6. **Tests** -- a `ScriptedTransport` in the always-compiled set covering the hop limit, the
   cross-host token drop, relative `Location` resolution, and `Range` answered 200.
7. **Binding** -- `HttpGetDelegate` becomes the transport shape above.

*Done when:* the cross-host token drop has a test that fails if the rule is removed, and both
presets are green.

## 8. Open

- **Bounded error-body capture.** Set at **4096 bytes** in `CurlHttpTransport`
  (`kMaximumCapturedErrorBytes`): enough for an error page's useful line, bounded enough that a
  hostile server cannot grow it. Revisit if a real message is ever truncated.
- **A delegate transport reports its headers late.** `IHttpTransport` gets a `HeadersCallback`
  fired before the body, which `CurlHttpTransport` drives from libcurl's header hook. A delegate
  cannot: the host returns everything at once, so `DelegateHttpTransport` fires it after the
  transfer. The consequence is that a progress callback in a hosted build sees `total == 0` while
  bytes arrive. Fixable by letting the host announce the response through the sink object before
  streaming; not worth the extra call shape until something needs the total.
- **Proxy configuration** is currently whatever libcurl picks up from the environment. Under a
  host transport it is whatever the host does. That difference is intended -- see section 2 -- but
  it is undocumented behaviour today.
