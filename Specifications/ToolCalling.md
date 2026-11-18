# Tool Calling

How a model's tool calls run through Mila::Agents: what the library provides and what an application brings.

**Status:** design, agreed 2026-10-25 (Todd), conceptually and with no release attached. Built to it at `+97`:
`Mila::Agents::Agent` (`Lib/Src/Agents/Agent.ixx`) with tools registered as a callable and a schema, the ask hook and its
three answers, no hook meaning deny, a call of a tool not offered or a tool that throws answered with an error result,
and a stopped reply running no waiting call. Built at `+98`: section 4's limits per response (`AgentLimits` --
rounds, time, reply tokens, none set by default), a stop or the time running out asking a running call to stop, and
a result the context cannot hold cut to what fits and the model told so; Chat and Mila Desktop's session run
their calls through the agent and hold no loop of their own. Built at `+99`: section 4's splice -- within a response
each round continues from the reply as generated and the results as the template frames them
(`Protocol::continueWithToolResults`), and Qwen continues from the state its last generation left. Built at
`+100`: sections 3.1 to 3.3 -- every call passes `ToolGate`: the arguments checked against the tool's parameters,
the person's rules (`ToolRules`), the question built from the tool as registered, the call's own time and a cap on
its result, and the record of each call on its event. Since 2026-10-26 an agent's tools can be withdrawn
(`withoutTools`), so one resident model serves conversations with different tools. Built at `+103`: section 3.4 on
Windows -- `ToolGrant`, given to every call allowed, narrowed to its tool's needs; off Windows a grant holds no
folder. An output schema and results other than text are not built. It replaces `ToolBroker.md` as the design of record for where tool calling lives; the code built to that
spec -- the broker in `Apps/Tools/Broker`, the client in `Mila/AI/`, the session's own handler map in Chat --
is refit to this one. Open decisions are in section 9.

---

## 1. The rule

**The Mila library provides everything a developer needs to build an application like Mila Desktop**
(`MilaDesktop.md`). An agent's tools are the core of that app, so tool calling is the library's. Leaving it to
each application would either give Mila Desktop a capability the library lacks (`Direction.md` 3.3) or require a
second harness library, which would be Mila under another name.

Two further reasons put it in the library and not in a harness across a wire. A tool result's tokens can be
appended to the live KV cache, with no re-render and no re-tokenize (section 4), only when the loop runs in the
process that holds the model. And how each model writes a call and expects a result is the model's own grammar.

There are two pieces: **`Mila::Agents`**, the agents API of the Mila library, a level on top of the model; and, at most
later, a process outside it (section 7).

**Through `Mila::Agents`, the application never runs a model's tool call itself.** It supplies tools and a way to ask
the person; Mila parses the call, checks it, runs it and puts the result back into the turn. An application that
wants to run calls its own way works at the model level (`Mila::Dnn` and the handle) and does so there: an explicit
choice to leave the structure, not a way around it.

## 2. Two levels

- **The model handle: the model's tool grammar.** Writing tool definitions into the prompt in the form the model
  was trained on; parsing calls out of the stream as the tokens arrive -- where a call starts and ends, its name and
  arguments, several calls in one turn, a call that is malformed or cut off; writing a result back in the model's
  own form; separating the stream into thinking, answer text and calls in progress, so an application can show text
  live and know when a call is forming. From outside it is family-blind: the caller sees a call, a name and
  arguments, never a family. Whether a model calls tools, or several at once, is a declared capability
  (`ModelHandle.md`). The inference server works at this level: it parses a call and returns it, since the outside
  harness owns its tools.
- **The agent, in `Mila::Agents`:** the loop of section 4 and the four pieces of section 3. Every call it parses runs
  through them.

## 3. The four pieces

### 3.1 Tool

- A name, a description, an input schema and, optionally, an output schema.
- **A summary, written for the person,** which is what a question shows. The description is written for the model.
- **Declared needs:** whether it reads files, writes files, uses the network or runs programs. They are facts the
  person is shown, the permissions of 3.3 decide on and a call's grant is narrowed to, not hints to the model. Which
  folders are touched is the grant's, not the tool's. *Running programs added 2026-10-27 (Todd,
  `ContainedPrograms.md`).*
- **The source is a field,** shown to the person. *Decided 2026-10-26 (Todd):* the name the model reads is letters,
  digits, `_` and `-`, since Anthropic's tool names refuse a dot; two tools of one name are refused when the second
  is registered, so two sources cannot collide.
- **The description is prompt text the model reads.** It is pinned when approved; a changed description is asked
  about again. A tool registered in-process cannot change: a name registers once on an agent. Pinning is needed
  when tools come from outside the application, and arrives with MCP (section 6).
- **Registration:** a C++ callable and its schema, registered on the agent.

### 3.2 Execution

How a call reaches the tool. **Initially in-process only:** a function of the application, called in its process.

- **The arguments are checked against the schema before anything runs.** The model's output is untrusted input.
  *Decided 2026-10-26 (Todd):* Mila checks a stated part of JSON Schema itself (`ArgumentSchema`): type,
  properties, required, additionalProperties, enum, const, items, and the length, range and item-count limits, at
  any depth. A schema with any other keyword is refused when the tool is registered, so no check is claimed that
  is not made. A call that fails is answered with an error naming what failed; no one is asked and nothing runs.
- A timeout, cancellation, and a cap on the result's size. *Decided 2026-10-26 (Todd):* a tool sets its own time
  and its own cap in tokens, else the response's (`AgentLimits` `tool_time`, `result_tokens`); none is set by
  default. A call out of its time is asked to stop through its `std::stop_token`, the model reads that it ran out,
  and the response goes on. A result over the cap is cut, by tokens, to the first that fit, with a note of how many
  of how many.
- **A failure reaches the model as an error result,** never as an exception into the application, so the model can
  recover.
- Results are text or structured data, and later images for models that see.

### 3.3 Permissions

A gate that every call passes; inside `Mila::Agents` there is no path to execution around it.

- **In:** the agent, the tool, the arguments and the tool's declared needs. **Out:** allow, ask or deny, and with
  allow a **grant** -- which folders, read-only or read-write, and whether the network -- and the call's time and
  size caps. *Decided 2026-10-27 (Todd):* the application gives the agent's session the most any call may reach
  (`Session::withGrant`, `Direction.md` 4.3); each call allowed is given that narrowed to its tool's needs -- no folder to a tool that declares
  no files, read-only to one that only reads, the network only to one that uses it -- so what a call reaches comes
  from the application's data, never the model's text. The caps stay with the gate, which enforces them (`+100`),
  rather than in the grant. The question shows the grant the call would be given.
- **Rules match on arguments,** not only on the tool: a read inside the project folder can be allowed while one
  outside it is denied. *Decided 2026-10-26 (Todd):* a rule names a tool, or a pattern of tools, and conditions on
  arguments named by JSON Pointer -- one of some values, or matching a pattern. Of the rules that apply, deny wins
  over ask and ask over allow; a call no rule applies to is asked about. A condition compares text, so a pattern on
  a path decides whether a call is asked about, never what the tool can reach: that is the grant's.
- **A call that will not run is never asked about:** a tool the model was not offered, or arguments that fail the
  schema.
- **Ask goes through a hook the application supplies.** Mila decides that the person must be asked and what they
  are shown: the tool's name and summary, every argument in full, and what it will touch. **The question is built
  from Mila's own data, never from text the model wrote.**
- **Answers:** once, for this conversation, or deny. Making a rule permanent is a change to the rules, never an
  answer to a question. An answer for the conversation stands in for asking again; it never overrides a deny rule.
- **No hook means no one to ask,** so ask is deny.
- **A denial reaches the model as a result** saying the person declined, so it can continue. A stopped reply
  cancels a pending question.
- **A record of every call** -- the decision, who decided, duration, sizes -- reaches the application as an event.
  Where it is kept is the application's.

### 3.4 Containment

What holds a call to its grant. Permissions decide; containment enforces. A grant is worth nothing unless the thing
running the tool holds it there.

**For now, containment is the grant object.** Mila passes the tool its grant as an object it acts through: the tool
opens files and makes connections through the grant, and the grant checks each at that moment. A check made ahead
of the tool, on the text of its arguments, is not enough, since what a name leads to is known only once it is
opened.

- **Files.** The grant accepts a path in one form only: a drive-letter path or a path relative to the granted
  folder -- the first, where several are granted -- each part an ordinary file or folder name: no `..`, no `:`, no
  reserved or control character, no name ending in a space or period, and no name Windows reserves for a device
  (`GrantPath`). It opens the file, takes the final path the operating system reports for it, and keeps it open
  only when that path lies within the granted folder's own final path. Short names, differences of case, symbolic
  links, junctions and mount points all lead to that final path, so the comparison decides for them. A hard link
  does not: which of a file's names is reported is not documented (Raymond Chen's *The Old New Thing*, 2016-04-06
  and 2019-04-10, give different answers), so *decided 2026-10-27 (Todd)* a file with more than one name on disk
  is refused. A new file or folder is created only in a folder already opened and checked, through the handle
  that folder was opened by, and never through a name that already exists, a link included; nothing is truncated
  before the file is checked.
- **Files on macOS** (designed 2026-11-17 from Apple's man pages on macOS 27.2.1 -- `open(2)`, `fcntl(2)`,
  `stat(2)`, `mkdir(2)` -- and the xnu source, measured the same day, built in 0.21.0-dev+152 as
  `Agents/Grants/GrantFileSystem.Apple.ixx`). The same rule by the platform's own calls, each on a file
  descriptor as Windows' are on a handle; every macOS Mila runs on is 27 or later (`CMakePresets.json`), so each call
  below exists on all of them.
  - *The path.* A path from the root (`/Users/...`) or relative to the granted folder, each part an ordinary name: no
    `..`, no control character, and not `..namedfork` in any case (which opens a file's resource fork, as `:` opens a
    stream on Windows). The names Windows reserves -- devices, a trailing space or period -- are ordinary names on
    macOS and are not refused there; nor are `:` and `\`, which are characters of a name, `/` alone separating names.
    The form is the platform's (`GrantPathForm`), and both forms are tested on every platform.
  - *Opening.* `open` and `openat` with `O_CLOEXEC`, so no program a tool later starts inherits the descriptor, and
    `O_NONBLOCK`, so a named pipe in the folder cannot hold the call; then `fstat`, and anything that is neither a
    regular file nor a folder -- a pipe, a socket, a device -- is refused.
  - *The final path.* `fcntl( F_GETPATH )` on the open descriptor, for the granted folder when the grant is made and
    for each file as it is opened; symbolic links, `/tmp` and `/var` (links into `/private`) and the firmlinks between
    the system and data volumes all lead to it. Kept only when it lies within a granted folder's own final path, as on
    Windows. A path longer than `MAXPATHLEN` cannot be reported, so it is refused, saying why.
  - *More than one name.* A file opened by its name in an open folder is opened with `O_UNIQUE`, which fails with
    `ENOTCAPABLE` when it has more than one hard link (xnu's own test, `tests/vfs/unique.c`), so the refusal is made in
    the same call as the open. One opened by its whole path is not, and `st_nlink` from `fstat` decides, as Windows'
    link count does; so a search that only asks what is at a path finds such a file on both platforms, and opening it
    is refused on both. A folder's `st_nlink` counts its subfolders, not its names, so a folder reports one.
  - *Creating.* A file by `openat( folder, name, O_CREAT | O_EXCL | O_NOFOLLOW )`, which fails when anything has the
    name, a symbolic link included (`open(2)`); a folder by `mkdirat( folder, name )`, then opened through the same
    folder with `O_DIRECTORY | O_NOFOLLOW` and checked by its final path before anything is made in it.
  - *Listing.* The open folder's own entries (`fdopendir` on a duplicate of its descriptor), each described by
    `fstatat` on that folder, so a link is listed as one and its target's kind is the target's.
  - *Not chosen:* `O_RESOLVE_BENEATH` as the rule. It refuses a link from one granted folder into another, which
    Windows accepts, so the two platforms would differ; the final path decides on both. `O_NOFOLLOW_ANY`, for the
    same reason.
  - *Measured 2026-11-17* on the development Mac, macOS 27.2.1, on its APFS volume and on case-sensitive APFS, HFS+
    and exFAT disk images (a probe outside the repository, `~/mila-work/grant-probe`):
    - `F_GETPATH` reports one path for a folder or file whichever way it was reached -- by `/Users` or by
      `/System/Volumes/Data/Users`, through a symbolic link, through `/tmp` (reported as `/private/tmp`) -- and in the
      case and Unicode form stored on disk, not the ones it was opened by, on every volume tried; so the comparison
      needs no normalising, and `F_GETPATH` is the call (`F_GETPATH_NOFIRMLINK` reports the data volume's form,
      equally single). A held file reports where it is now after its folder is renamed or it is moved out, and its
      last path after it is deleted, so the check is made as the file is opened, as on Windows.
    - Past `MAXPATHLEN` (1,024 bytes) `F_GETPATH` fails with `ENOSPC` and an open by the whole path with
      `ENAMETOOLONG`, so such a file is refused, saying its path is longer than macOS can report. This is the
      platform's own limit; Windows has none so low.
    - `O_UNIQUE` refuses a file with two names with `ENOTCAPABLE` on APFS, case-sensitive APFS and HFS+, and opens a
      folder however many subfolders it has; exFAT makes no second name (`link` fails with `ENOTSUP`). `link` on a
      folder is refused (`EPERM`) on every volume, HFS+ included, so a person cannot make a folder with two names, and
      HFS+ needs no refusal. Only the system makes such folders, in Time Machine backups written to HFS+, which
      `O_UNIQUE` does not see; a grant of a folder inside one is a known gap, stated here.
    - A named pipe opened with `O_NONBLOCK` returns at once and `fstat` reports it a pipe, so it is refused without
      waiting; a socket file fails to open (`EOPNOTSUPP`).
    - Writing `file/..namedfork/rsrc` makes a resource fork, reported as `.../file/..namedfork/rsrc`; on exFAT it
      makes a hidden `.vfs_rsrc_stream_...` file in the folder. So the path form refuses `..namedfork`.
    - `O_CREAT | O_EXCL` refuses a name that is a dangling symbolic link with `EEXIST`, with `O_NOFOLLOW` or without,
      and makes nothing where the link points; a folder swapped for a link before it is opened with
      `O_DIRECTORY | O_NOFOLLOW` is refused with `ENOTDIR` (not `ELOOP`), so both errors read "is not a folder".
    - A listing gives a link `DT_LNK`; `fstatat` on the folder reports the link itself, and, followed, its target's
      kind, or `ENOENT` for a link to nothing, which is listed as a link and as no folder.
    - `O_RESOLVE_BENEATH` refuses a link out of the folder (`ENOTCAPABLE`), as *Not chosen* expected.
  - *Privacy consent, measured 2026-11-17* with Todd at the Mac answering macOS's questions: a child process of an
    Electron app, as `mila-session` is of Mila Assistant's window, opening a folder macOS protects
    (`~/mila-work/grant-probe/consent.sh`).
    - Opened with no open panel, Downloads and Desktop each brought macOS's question, the open waiting on the answer
      (3.6 s and 2.8 s). Refused, the open fails with `EPERM` -- not `ENOENT` -- and fails at once from then on; allowed,
      it opens at once from then on. So a grant reports `EPERM` from macOS as its own refusal, naming System Settings'
      *Privacy & Security* as where it is changed, and never as a missing file.
    - A folder the person chose in the app's own open panel, inside Documents, was opened by the child at once with no
      question; Documents itself, opened by the child next, waited 7.0 s, macOS's question shown and answered. So the
      open panel gives the app that folder and nothing around it, which is what a grant gives.
    - A mounted disk image was opened at once with no question, so it stands for no removable drive in macOS's
      consent; a real external drive is not measured.
    - **Not measured:** whether a folder chosen in the open panel stays open to the app's child after the app is
      quit and started again, which a chat reopened with its folder needs (`Apps/AssistantInterface.md` 5.5); a real
      external drive; iCloud Drive and network volumes, which the development Mac does not have.
  - *Tests:* `Lib/Tests/Agents/ToolGrant.Files.cpp` and `FileTools.cpp` run on both platforms, each refused access
    beside the same access allowed, with each platform's own way to it -- a link to a folder outside (a junction on
    Windows, a symbolic link on macOS), a symbolic link to a file outside, a hard link to a file outside, a case other
    than the one on disk, a file created through a name that is a link, a folder replaced by a link once the grant has
    used it -- and on macOS also a path through `/tmp` and through `/var`, a named pipe, `..namedfork`, and a path past
    what `F_GETPATH` can report. `GrantPath.cpp` tests both forms on every platform. Passed 2026-11-17 at
    0.21.0-dev+152: 129 of 129 grant, file-tool, gate and program tests in `x64-claude-verify` (MXC 1.1.0 named), and
    98 of 98 grant, file-tool, gate and session tests on the development Mac, none skipped on either.
- **Files on Linux** (designed 2026-11-17 from the Linux man pages 6.17 -- `open(2)`, `openat2(2)`, `readlink(2)`,
  `proc_pid_fd(5)`, `stat(2)`, `mkdir(2)`, `link(2)`, `readdir(3)` -- and measured the same day in WSL, kernel 6.18,
  as `Agents/Grants/GrantFileSystem.Linux.ixx`). The same rule by Linux's own calls, each on a file descriptor as on
  macOS.
  - *The path.* macOS's form (`GrantPathForm::Posix`): a path from `/` or relative to the granted folder, each part an
    ordinary name, no `..`, no control character. `..namedfork` is refused as on macOS, since the form is one; on
    Linux it is only an unusual name.
  - *Opening.* Every open is `openat2` with `RESOLVE_NO_MAGICLINKS` (Linux 5.6), with `O_CLOEXEC`, `O_NONBLOCK` and
    `O_NOCTTY`; then `fstat`, and anything that is neither a regular file nor a folder is refused, as on macOS. A
    *magic link* -- `/proc/<pid>/fd/<n>`, `/proc/<pid>/cwd`, `/proc/<pid>/root`, and `/dev/stdin` through them -- is
    not a path at all but a reference to something a process holds open, which may be a file deleted, outside the
    process's root, or in another program's mount namespace (`openat2(2)`); a grant never follows one, at the end of a
    path or within it, and the refusal says so. Windows and macOS have no such link, so this refuses nothing there. A
    kernel older than 5.6, or a sandbox that refuses `openat2`, fails every open, saying why; no older call is used
    in its place.
  - *The final path.* `readlink` on `/proc/self/fd/<n>`, the path the kernel keeps for the open descriptor, every
    symbolic link resolved (`proc_pid_fd(5)`). Kept only when it lies within a granted folder's path, as on the other
    two. Without `/proc` mounted nothing can be checked, so every open is refused, saying why. Paths past 4,096 bytes
    cannot be reported (`ENAMETOOLONG`), and such a file is refused, saying so, as macOS refuses past its 1,024.
  - *Case.* On a volume that ignores case -- a Windows drive in WSL, FAT and exFAT, an ext4 folder made
    case-insensitive -- `/proc` reports each name in the case it was looked up by, not the one stored on disk
    (measured), where Windows and macOS report the stored case. So a granted folder's path is taken in the names the
    volume stores (`onDiskPath`): each folder up to `/` found by its device and inode in its parent's listing, which
    crosses a mount as `..` does. A file whose reported path lies outside every granted folder is looked at the same
    way before it is refused, so a path that differs from the folder's only by case is accepted, as on the other two.
    The reported path is always compared first: equal names lead to the same folders whether the volume heeds case or
    not, so it never lets in what the stored names would keep out, and the listing walk, which costs a listing per
    folder up to `/`, is made only for what would otherwise be refused. A folder on the way up that cannot be listed
    leaves the file refused. On Windows and macOS `onDiskPath` is the final path.
  - *More than one name.* `st_nlink` from `fstat`, read after every open, as Windows' link count is; Linux has no
    `O_UNIQUE`. `link` on a folder is refused (`EPERM`) on every file system tried, so a folder reports one, its own
    count being its subfolders'.
  - *Creating.* A file by `O_CREAT | O_EXCL | O_NOFOLLOW` in a folder already opened, which fails with `EEXIST` when
    anything has the name, a link to nothing included, and makes nothing where the link points; a folder by `mkdirat`,
    then opened through the same folder with `O_DIRECTORY | O_NOFOLLOW`, which a link swapped in fails with `ENOTDIR`.
  - *Listing.* As on macOS: `fdopendir` on a duplicate, each name described by `fstatat` on the folder.
  - *Mounts.* A file reached through a mount -- a bind mount, a FUSE mount, another volume -- inside a granted folder is
    reported by the path it was reached through, so it lies inside, as a volume mounted inside a granted folder does on
    macOS. Only an administrator, or the owner of a mount namespace Mila is not in, can make one.
  - *Not chosen:* `RESOLVE_BENEATH` and `RESOLVE_IN_ROOT` as the rule, which would refuse a link from one granted folder
    into another, accepted on the other two; `RESOLVE_NO_XDEV`, which would refuse a mount inside a granted folder,
    accepted on macOS; opening with `O_PATH` and opening again through `/proc` once checked, so that a device is never
    opened before it is refused -- a link to a device opens it before `fstat` refuses it on all three platforms alike,
    so that is a shared question, filed, not a Linux one; and comparing every path in a case-blind way, which would
    let a folder named `PROJECT` pass for a granted `project` on a volume that heeds case.
  - *Measured 2026-11-17* in WSL (Ubuntu 26.04, kernel 6.18.58) on tmpfs (`/tmp`), ext4 (the home folder) and the
    Windows drive (`/mnt/d`, 9p), with a probe outside the repository (`D:\Claude\linuxgrant\probe`):
    - `/proc/self/fd` reports one path for a file or folder however it was reached -- through a symbolic link out or
      in, with `./` and doubled slashes -- on all three; a file still open after it is deleted reports its last path
      and ` (deleted)`, and a folder renamed while open reports its new path, so the check is made as the file is
      opened, as on the other two.
    - On `/mnt/d`, `PROJECT/INSIDE.TXT` opened `project/inside.txt` and `/proc` reported it in upper case; tmpfs and
      ext4 refused the name, as volumes that heed case do.
    - A second name made by `link` is reported as itself, `st_nlink` 2, on all three; `link` on a folder fails with
      `EPERM` on all three.
    - A named pipe opened with `O_NONBLOCK`, to read or to read and write, opens at once and `fstat` reports it a pipe;
      a socket file fails to open with `ENXIO`; the Windows drive makes neither (`EOPNOTSUPP`).
    - `O_CREAT | O_EXCL | O_NOFOLLOW` through a link to nothing fails with `EEXIST` and makes nothing outside; a folder
      swapped for a link fails `O_DIRECTORY | O_NOFOLLOW` with `ENOTDIR`, on all three.
    - With `RESOLVE_NO_MAGICLINKS`, a link in the folder to `/proc/self/fd/<n>`, that path itself, a link to
      `/proc/self/cwd`, and `/dev/stdin` each fail with `ELOOP`; an ordinary link in or out, `/proc/self/status`, and
      creating a file open as before. Without it, the link to `/proc/self/fd/<n>` opened the outside file it held.
    - `/proc` reports paths up to about 4,060 bytes and fails with `ENAMETOOLONG` past 4,096; a file there can still be
      made through its folder, and is then refused as unreportable; an open by its whole path fails the same way.
    - A file reached through a bind mount, made in a user namespace with `unshare -rm`, is reported by the path through
      the mount.
  - *Tests:* `ToolGrant.Files.cpp` and `FileTools.cpp` run on Linux as on the other two, with a symbolic link for a
    folder link; on Linux also a link through `/proc` to an open file outside beside a link to a file inside, a named
    pipe, a socket file, and a path past what `/proc` can report beside one within it; and on every platform a folder
    whose name differs from the granted one's only by case, refused where the volume heeds case, beside the same path
    inside in another case, allowed where it does not. Linux runs them in WSL, on tmpfs and on the Windows drive.
    Passed 2026-11-17 at 0.21.0-dev+153, the grant, file-tool, path and gate tests: in WSL (`linux-clang-cpu-release`,
    Clang 22) 98 on tmpfs, the case-ignoring test skipped, and 95 on the Windows drive, which makes no pipe or socket
    and ignores case, so the path in another case is allowed there through the stored names; 98 on Windows and 99 on
    the development Mac, each skipping the case-heeding test its temporary folder cannot run.
- **Network.** The grant connects only to public addresses. *Decided 2026-10-27 (Todd):* public is globally
  reachable as IANA's IPv4 and IPv6 special-purpose registries record it, multicast refused too, and an IPv6
  address that carries an IPv4 one -- IPv4-mapped, NAT64, 6to4 -- is public only when that IPv4 address is
  (`PublicAddress`); without that last rule `::ffff:127.0.0.1` would pass. The check is made on the address each
  connection is about to be made to, once the name is resolved, so the decision is on the connection, not on the
  name. A redirect is a new connection and is checked the same way. No proxy is used, one named in the
  environment included, since the connection checked would be the proxy's.
- **Where.** *Decided 2026-10-27 (Todd):* Windows only for now. *Decided 2026-11-15 (Todd):* macOS too, in
  v0.21.0, macOS being a platform at parity with Windows (`BACKLOG.md`, *Mila on Apple Silicon*), by the design under
  *Files on macOS* above, built in 0.21.0-dev+152. *Decided 2026-11-17 (Todd):* Linux too, at parity with the other
  two (`CLAUDE.md`), by the design under *Files on Linux* above, built in 0.21.0-dev+153 and tested in WSL, where
  Mila Assistant's window cannot run, so the window on Linux is unverified. `Agents.GrantFileSystem` has four
  candidate files: Windows', macOS's (`.Apple`), Linux's (`.Linux`) and one that refuses every call (`.Null`),
  compiled on any other system. Each gives the path's form (`GrantPathForm`), where a full path starts (`rootPath`:
  `\\?\D:\` on Windows, `/` elsewhere), the final path and the path in stored names (`onDiskPath`), so `ToolGrant` is
  the same code on all three.
- **What it holds.** A tool that works through its grant is held to it. One that opens files directly is not; an
  in-process tool is the application's own code, and that is the application's choice.
- **Programs.** *Decided 2026-10-27 (Todd):* the grant also runs a program for a tool that declares it runs
  programs, through a container the grant holds, as it holds its transport (`ContainedPrograms.md`). The program
  and everything it starts reach the grant's folders, and the network as the grant gives it; no other of the
  person's files by any name, nor their windows, screen, clipboard or input. Much of their registry under `HKCU` is
  read, and nothing in the policy refuses it; that is a known gap, stated where the person turns the container on
  (`ContainedPrograms.md` 4.9). On Windows the
  container is Microsoft's MXC, experimental in v0.21.0 and off by default; a grant without one refuses to run a
  program. MXC decides by the name a file is opened by, so before each run a file with a name outside the grant is
  withheld from it; a name made from outside the container while a run is going is a known gap.
- **Mila never claims containment it does not enforce.** A tool that would need more -- a shell, another party's
  server, code an agent wrote -- is not offered until an implementation exists that enforces its whole grant. The
  container is that implementation for a shell, and only where it is on.

The checks are the hard, security-critical part, so they are Mila's and not each application's: written once, and
each tested by an access the grant refuses beside the same kind of access inside the grant, allowed. *Decided
2026-10-26 (Todd):* the tests do not also run each access with its check removed; that step belongs to the
operating-system sandbox's own measurement (`Vnext.md`), not to the grant's.

## 4. The loop and the splice

- Generate; when a call completes, pause, run it through section 3, append the result's tokens to the live cache,
  and continue the same turn, until the model gives its answer.
- **Limits per turn:** the most tool rounds, the total time, the reply budget. Stopping a reply cancels a call that
  is waiting or running. *Decided 2026-10-25 (Todd):* the library sets none by default -- the context bounds a
  response that has none -- and an application sets its own; Chat keeps the four rounds it has always run, except
  with the tools Mila::Agents provides, whose work takes more, where it sets none, as Build & Code does; a number
  no measurement set. The reply budget counts every round's tokens. The time the person takes to answer a question
  is not the response's. The calls of the last round allowed are answered as not run, since no round would read
  their results. A running call is asked to stop through a `std::stop_token`; an in-process tool can only
  cooperate, and what it returns after the stop is not given to the model.
- **A result larger than the context left** is cut or refused, and the model told which; it never silently
  overflows. *Decided 2026-10-25 (Todd):* cut, by tokens, to the first that fit, with a note of how many of how
  many; refused only when none fit. The cut keeps the prompt inside the context and reserves nothing for the
  answer, so the result cap of 3.2 is what keeps an ordinary result small.
- **The splice.** *Decided 2026-10-25 (Todd):* within a response the model continues from its own tokens, not from
  a rendering of them. A round's prompt is the last round's prompt and its reply as generated, followed by the
  results as the template frames them after a call -- the model's protocol gives that text, tested against the
  template's full rendering -- so a round prefills only the results it adds. A reply whose last call is left open is
  rendered instead, so the model never reads a call it did not finish followed by its result. Across responses the
  conversation is rendered: Gemma and Llama prefill from where the rendering first differs from what they hold,
  such as the reasoning a template drops, and Qwen from the end of its last reply when the rendering extends it,
  else from the end of the last round's prompt. Measured in `Direction.md` 5.3. The model ends a reply on the token the template opens its results with (Gemma's
  `<|tool_response>`, Qwen's `<|im_end|>`, Llama's `<|eot_id|>`), and decode-ahead has already put that token in the
  cache, so the results' first token is never prefilled.
- **Prefix reuse is the model's property**, read from the handle: a rewind to any position, or for a model with
  recurrent state a return to a position it saved (`Direction.md` 5.3); Qwen also continues from the state its last
  generation left when a prompt extends every token that state holds. The loop does not know which family it drives.
- **The conversation is the agent's;** its history and the cache always agree.

## 5. What the application brings

- **Which tools it offers, and any of its own.** The models API has none. *Decided 2026-11-01 (Todd), replacing "the tools are
  all the application's" of 2026-10-25:* Mila::Agents provides the tools two applications offer, since neither may be built
  from the other's code: the six file tools (`FileTools`) and reading a web page (`WebPageReader`), each carrying its
  name, description, schema, summary and needs, each written through its grant. Offering one is the application's
  act: a prompt file names it, and the application gives the folder or network it works through. A tool only one
  application has stays its own: Mila Desktop's command, plan and web search tools. A tool moves into Mila::Agents when a
  second application needs it.
- **Its default rules,** and where the person's rules are kept and how they are edited: a file, a settings page, a
  command.
- **How a question looks:** a window, a console line, or nothing for a headless program.
- **The project folder,** chosen with the person, which becomes the files grant.
- **Where conversations are saved.** Mila gives a conversation as data it can restore.

## 6. Later

- **MCP, client and server** -- not initially. A client would add local servers started as processes over stdio
  and remote servers over HTTP, whose tools join the same registry, so the loop and the gate treat them like any
  other; everything a server says is untrusted, and a changed tool list withdraws its approvals. A server would
  let an application offer its tools to other agents.
- **Tools in a process of their own** arrive with the client: such a tool is an MCP server Mila starts over stdio.
  The approval round trip MCP specifies for one -- its input-required result -- is `ToolBroker.md` section 5.
- **A sandbox** under the same containment interface, enforcing a grant whole. *Reversed 2026-10-27 (Todd):*
  Microsoft's execution containers, kept out on 2026-10-25, come into v0.21.0 as an experiment, a program run
  through the grant (3.4, `ContainedPrograms.md`); MCP servers and code an agent writes run through it later.
- **Remote servers' sign-in tokens,** held where neither the model nor the agent's text can reach them.

## 7. The broker

A per-user broker is not a fifth piece. It is sections 3.2 to 3.4 hosted in a process of their own, which makes it
an application built on the library, and it is needed only when the rules must hold against the application
itself -- one set of rules over agents someone else wrote, under an account those agents cannot control. Rules
inside the application protect the person from the model, steered by a hostile page or file; they cannot protect
the person from the application, which can skip its own. `ToolBroker.md` records the broker as designed on
2026-10-24.

## 8. Layering

The tool pieces need nothing of the model, and the model needs nothing of them; only the loop joins them. A program
that wants tools alone pulls in no CUDA. What the pieces add is JSON for schemas and arguments and the operating
system's file and socket calls for the grant; HTTP, TLS and process launching come only with section 6.

## 9. Open decisions

1. **Several calls in one turn:** run together or in order. Reads and writes may need different answers.
2. **Rules after untrusted content.** Once a fetched page or a file the person did not write enters the
   conversation, whether rules tighten -- for example, ask before any write that follows. This is now the centre of
   `Direction/MilaVision.md` section 8, whose proposals are not yet decided.
3. **Connections through the grant.** *Decided 2026-10-27 (Todd):* a tool fetches through the grant's own
   transport (`ToolGrant::transport`), libcurl with each connection's address checked before it is made, so every
   connection, a redirect's included, is checked. A transport the application supplies, such as the Python
   binding's, is never a grant's.
4. **Where the agents API sits in the build.** *Decided 2026-11-03 (Todd), replacing a target of its own of 2026-11-02 and
   a library of its own of 2026-10-31: `Lib/Src/Agents/`, built in the Mila target and brought by `import Mila;`*
   (`Direction.md` decision 2). Still open: whether the tool pieces become a target of their own, so a
   program wanting tools alone needs no GPU stack -- asked when such a program exists.
5. **The record of a call.** *Decided 2026-10-26 (Todd):* the event of each call answered carries the decision and
   who made it (`ToolDecision`), how long the tool ran and the size of what it returned; it carries the call, so
   the arguments whole. Whether they are kept whole is the application's, which keeps the record.

## 10. Relationship to other specs

- `Direction.md` 3 and 8 -- the layering and decision 6, which point here.
- `Direction/MilaVision.md` -- the local promise and what content may cause, both of which this gate and grant must hold.
- `ModelHandle.md` -- the handle the grammar of section 2 lives on, and its declared capabilities.
- `MilaDesktop.md` 4.5 -- Mila Desktop's question and answers, which this design's hook presents.
- `ToolBroker.md` -- the earlier design; its approval round trip informs section 6, its checks are section 3.4's.
- `Notebooks/LocalToolService.md` -- the reading behind both: placement, identity, secrets, the round trip
  measured, and prior art.
