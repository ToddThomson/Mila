# Contained Programs

How a tool runs a program held to its grant, through Microsoft eXecution Container (MXC), and the first tool that
does: the Mila app's shell in Build & Code.

**Status:** agreed 2026-10-27 (Todd, decisions 1 to 6 of section 9); phase 1 measured the same day; decisions 7
to 10 agreed 2026-10-31, and phase 2, the library, built that day; the tool (phase 3) not begun. Decided before it (Todd, 2026-10-27): MXC goes into
tool calling for v0.21.0 as an experiment, behind a switch that is off by default, as long as the grant
(`ToolCalling.md` 3.4) remains v0.21.0's safe tool calling; the design works around the two accesses MXC's policy
accepts that no grant gives; a file that gains a second name from outside while a run is going is accepted as a
known gap. Evidence: `Notebooks/MxcConformance.md`. Decisions in section 9.

---

## 1. What it is for

`ToolCalling.md` 3.4 keeps a shell, other people's tool servers and code an agent writes off the menu until
something enforces a whole grant around them, since such a tool opens files itself rather than through the grant.
MXC runs one process, and everything it starts, under one policy: folders read-write, read-only or denied, the
network, the UI. This design puts that under the grant, so a tool asks its grant to run a program as it asks it to
open a file, and the program reaches what the grant gives and nothing else.

The first use is a tool that runs a command in Build & Code's folder: build, test, `git`. Third-party tool servers
and code an agent writes come later through the same call (section 8).

## 2. What MXC holds, and its gaps

Measured on v1.0.1 (`Notebooks/MxcConformance.md`), on Windows and Linux alike unless named:

- **Holds:** files outside the granted folders, through any name that leads there -- links, junctions, UNC and `\\?\`
  forms -- refused; read-only and denied folders as declared; child processes, and on Windows processes started
  through WMI, COM and Task Scheduler, under the same policy; nothing outlives the run, and killing the host ends it;
  no network at all, name resolution included; on Windows, public addresses only through a deny list of IANA's
  special-purpose ranges; a fresh container each run; the host's environment filtered.
- **Gap 1, files with more than one name.** The policy decides by the name used to open a file. A file with a name
  inside a granted folder and another outside it, or in a denied folder, is accepted through its inside name. A
  contained process cannot make such a name to a file outside; something outside the container can.
- **Gap 2, the screen.** With the UI disabled, most build tools do not start. With it on, every isolation mode
  accepts a read of the person's screen, and no policy field refuses it.

- **Gap 3, the registry** (found in phase 1). The policy has no registry field; a program reads much of the
  person's `HKCU`, and writes none of it.

Sections 4.3 and 4.4 close the first two; the third is a known gap, stated where the setting is (4.9, decision 8).

## 3. Where it sits

```
Tool.run( arguments, grant, stop )
    grant.runProgram( request, stop )          ToolGrant: refuses unless the grant may run programs
        ProgramContainer                       the interface; the grant holds one or none
            MxcProgramContainer                Windows: mxc_ffi.dll, loaded at run time
                one desktop of Mila's own      section 4.4
```

The container is the grant's, as its transport is (`ToolCalling.md` 9.3): a tool reaches a program only through
the grant its call was given, so the gate's decision and the narrowing to the tool's needs (3.3) apply unchanged. A
grant made with no container cannot run a program, and says so.

## 4. The design

### 4.1 The grant runs programs

- **`ToolNeeds::runs_programs`**, beside reading files, writing files and the network: shown to the person, and the
  grant's container goes only to a tool that declares it (`narrowedTo`).
- **`ToolGrant::runProgram( const ProgramRequest&, std::stop_token ) -> ProgramResult`.** The request: the command
  line, the folder it starts in (a folder of the grant), its time, and the most output it returns. The result: the
  exit code, standard output and standard error each cut to the cap with a note of how much was dropped, whether
  the time ran out, whether it was stopped, and which files the run was not given (4.3).
- **The grant decides the policy, never the tool.** The tool names the command; which folders, how, and whether the
  network, are the grant's (4.2). A request cannot widen any of it.
- **A grant with no container refuses** with "this tool was granted no way to run programs", as one with no network
  refuses a fetch.

### 4.2 From the grant to MXC's policy

| Grant | MXC request |
|---|---|
| Folders granted read-write | `filesystem.readwritePaths`, each the folder's final path, taken when the grant was made |
| Folders granted read-only | `filesystem.readonlyPaths` |
| Program folders (4.6) | `filesystem.readonlyPaths` |
| Files with more than one name (4.3) | `filesystem.deniedPaths`, for this run |
| No network | `network.egress.default: deny` |
| The network | `egress.default: allow`, with a deny rule holding every range `PublicAddress` refuses (4.5) |
| -- | `network.ingress.default: deny`, `hostLoopback: deny` |
| -- | `ui.disable: false`, `clipboard: none`, `injection: false`, on a desktop of Mila's own (4.4) |
| -- | `lifecycle.destroyOnExit: true`; `process.timeout`: the call's time, which Mila also keeps itself (4.8) |
| -- | `process.env`: the environment of 4.7, never MXC's default |
| -- | `telemetry.enabled: false` (4.10) |
| -- | `containment: processcontainer`; never `leastPrivilege`, under which the build tools do not start |

### 4.3 Files with more than one name

The granted folders are walked once, before the first run, and from then on watched (`ReadDirectoryChangesW`, the
whole tree, one watcher per folder); before each later run only the files the watcher has named since are read
again. That holds on ReFS; on NTFS, and any other file system, the folders are walked whole before every run,
since the watcher there misses a name made outside (below). Each file whose link count is more than one has its names listed (`FindFirstFileNameW` /
`FindNextFileNameW`). A file whose every name lies within a granted folder that gives at least the same access is
left alone; any other is put in `deniedPaths` for that run, by its name inside, and the result names it so the model
knows why it is unreadable. When a watcher overflows and so names nothing, its folder is walked again, and a run
whose walk passes a bound is refused, saying why (decision 7).

- **Why deny the file rather than refuse the run:** ordinary tools make such names. pnpm links `node_modules` to its
  store, and `git clone` from a folder on the same drive links its objects to the source's. Refusing the run would
  refuse those projects whole; denying the file withholds only what lies outside.
- **The known gap:** a name made from outside the container after the walk, while the run is going, is not seen.
  A contained process cannot make one to a file outside (refused, measured).
- **Cost, measured in phase 1:** 7,484 files take about 115 ms, the repository with its build directories --
  357,961 files -- 5.5 s, the link count costing the same read by name or through an opened file. A full walk before
  every command is too slow for a project of that size, so the walk is paid once and the watcher keeps it current.
- **What the watcher names, measured 2026-10-31:** a name made inside for a file outside, as an added name; a name
  made outside for a file inside, as a change to that file, but only when the watcher asks for attribute changes
  (`FILE_NOTIFY_CHANGE_ATTRIBUTES`), so it always does -- on ReFS only. On NTFS a name made outside for a file
  inside is not reported at all (measured by phase 2's tests), so on any file system but ReFS the folder is walked
  whole before every run (decision 7). An outside name removed is not named, so the files withheld
  from the last run are read again before each run with those the watcher named.
- **Its load, measured 2026-10-31:** with a 64 KB buffer nothing was lost writing 20,000 files in under a second,
  installing the Mila app's npm packages, or rebuilding every Mila target with the whole repository watched
  (18,123 events in 350 s); a 4 KB buffer overflowed 19 times in that build. So a walk after the first is rare.
- **The bound** (Todd, 2026-10-31): no count of files. A walk, the first or one after an overflow,
  is part of the run and counts against the call's own time; a walk that does not finish within it refuses the run,
  saying the folder is too large to check in the time given.
- **Linux** has no call that lists a file's names, so a file with more than one name is denied there whatever its
  other names. The grant is Windows only (3.4), so this is for later.

### 4.4 A desktop of Mila's own

The contained process runs on a desktop in `WinSta0` that Mila creates for it, never on the person's. Its DACL
grants SYSTEM, the person, *ALL APPLICATION PACKAGES* and *ALL RESTRICTED APPLICATION PACKAGES*, and it is labelled
low integrity -- the access `user32.dll` needs to start in an AppContainer (microsoft/mxc#1465). Measured: the build
tools start; reading the screen, seeing the person's windows, reaching the taskbar, the clipboard and injected
input are refused.

The run reaches that desktop through the thread that starts it: the container's thread in `mila-session` sets the
desktop (`SetThreadDesktop`) before `mxc_spawn_json`, and MXC's process lands on it (measured in phase 1; decision
3). The thread does nothing else, since a thread that owns a window or a hook cannot change its desktop.

### 4.5 The network

The grant's network is public addresses only (3.4). MXC filters by address, so the grant's rule becomes a deny list:
every IANA special-purpose range, IPv4 and IPv6, and multicast, generated from `PublicAddress`'s own table so the
two cannot drift. The IPv6 forms that carry an IPv4 address -- IPv4-mapped, NAT64, 6to4 -- are denied whole, which
is stricter than the grant, which decides by the IPv4 address inside. Measured on Windows: public HTTPS accepted; the
LAN gateway, loopback and loopback written as `::ffff:127.0.0.1` refused. Names still resolve under that policy,
though the LAN's resolver is in a denied range; MXC does not document how, and with no network they do not.
Loopback is refused by MXC's own host rule as well, so it cannot show that the list is sent; the LAN gateway is
refused by the list alone, and its test is the one that fails without it.

### 4.6 Program folders

A command runs programs that live outside the project: Git, CMake, Visual Studio's compiler, Node. The application
names those folders, read-only, as part of the grant it gives the session (`Session::withGrant`), and the person is
shown them. They are never the model's to choose. Measured in phase 1: with no folder granted, a file under
`C:\Program Files` is read by its path while the folder cannot be listed, so granting the program folders is what
lets the tools list and search their own; `C:\ProgramData`, `C:\Users` and other drives are refused.

### 4.7 The environment

A run is given the environment Mila writes (`process.env`), never MXC's default, which is the person's own user
and system variables -- whatever a person keeps there, a token included. The list: `SystemRoot`, `windir`,
`ComSpec`, `PATHEXT`, `PATH` (System32, Windows PowerShell and the program folders), and `TEMP`, `TMP` and
`LOCALAPPDATA` pointing into the project's `.mila` folder, `LOCALAPPDATA` being one MXC refuses to start without.
The project is the first folder granted to write; a grant with none refuses to run a program.
Measured: `cmd`, `git status`, `cl`, `node` and PowerShell start with these nine.

### 4.8 Time, stop and output

`mxc_spawn_json` starts the run and returns a handle; standard output and error are each read on a thread of their
own and kept to the cap, the rest read and dropped so the program never blocks on a full pipe; the container's
thread polls `mxc_sandbox_try_wait` and, on the call's `std::stop_token` or its time, ends the run from that same
thread (`mxc_sandbox_kill`, `mxc_sandbox_kill_for_timeout`), as MXC's concurrency contract asks. Polling does not
honour the request's own timeout, so the time is Mila's to keep (phase 1). Measured: a stop at 2,000 ms ends the
run at 2,040 ms with nothing left; 1.2 MB of output kept to a 4 KB cap. `mxc_run_json` holds the output whole and
cannot be stopped, so it is not used.

### 4.9 What the container leaves readable

Measured in phase 1, with no folder granted: the registry is not in MXC's policy. `HKLM` is read; under `HKCU`,
`Environment`, `Software` and its keys, `Software\Classes`, Explorer's shell folders and the desktop settings are
read, while `Volatile Environment`, `Console` and the Run key are refused, and nothing under `HKCU` can be written.
The person's saved credentials are refused (`cmdkey`, the Windows Credentials vault). So a program run through the
grant reads what the person keeps under `HKCU`, which no grant gives. It is a known gap, stated where the setting is
(section 6) and reported to Microsoft with the other two (decision 8).

### 4.10 The library behind it

- **`mxc_ffi.dll` is loaded at run time** from the application's folder, never linked, so a build without it still
  runs and a machine without it offers no program-running tool. The Mila app's installer carries the v1.0.1 file
  (MIT, about 6 MB) beside `mila-session`.
- **The version is pinned.** MXC publishes no C header (microsoft/mxc#1449), so Mila mirrors the `repr(C)` layouts of
  one version, checks `mxc_version()` on load, and refuses any other.
- **Availability is asked, not assumed:** `mxc_platform_support_json` at load; on a Windows build older than
  26300.9550 the container is absent and so is the tool.
- **Telemetry stays off.** Microsoft's builds of MXC can send diagnostic events; one is sent only when the run's
  request enables it, the person has consented through MXC's own prompt, and administrative policy allows it
  (MXC's `docs/telemetry.md`). Mila writes `telemetry.enabled: false` in every request and never shows MXC's prompt,
  so another application's consent changes nothing for Mila's runs. Phase 1: consent `undetermined`, and no
  connection owned by the host during a run.

### 4.11 The switch

- **The build:** a CMake option, off by default, selects `MxcProgramContainer`'s module or a module whose container
  is always absent -- two candidate files for one module, as the grant's file system has (no `#ifdef` in a module).
  Off, nothing of MXC is built or shipped.
- **The app:** a setting, off by default, labelled experimental. Off, the shell tool is not registered.

## 5. The first tool: running a command in Build & Code

- **`run_command`**, the Mila app's own (Mila ships no tools): arguments `command` and an optional time; needs: reads
  and writes files, runs programs, no network unless the person turns it on for the conversation.
- **Always asked.** No default rule allows it, and the question shows the command whole, the folder, the program
  folders and whether the network.
- The command runs under Windows PowerShell 5.1, present on every Windows 11 (decision 4). `cmd`'s `dir` is refused
  in a granted folder (phase 1), so the tool's description names PowerShell's own commands.
- The result the model reads: the exit code, the output as cut, and any files withheld (4.3).

## 6. What the person is told

The setting says what it does and what it does not: commands run in a Windows container limited to the project
folder and the program folders shown, on a desktop of their own, with no network unless allowed; experimental; on
Windows 11 26H2 or later. It never calls the tool sandboxed. The known gaps -- a file given a second name during a
run (4.3), and the registry under `HKCU` read (4.9) -- are stated where the setting is.

## 7. Modules

One type each, in `Mila/Src/AI/` unless named:

| Module | Type |
|---|---|
| `AI.ProgramRequest` | `ProgramRequest` |
| `AI.ProgramResult` | `ProgramResult` |
| `AI.ContainedRun` | `ContainedRun`, what the grant hands its container: the command and every folder by its final path |
| `AI.ProgramContainer` | `ProgramContainer`, the interface |
| `AI.MxcProgramContainer` | the MXC implementation, and its always-absent twin, two candidate files |
| `AI.ToolDesktop` | the desktop of 4.4, created once and closed with the container |
| `AI.SecondNameWatch` | the walk and watcher of 4.3 for one granted folder: the files with a name elsewhere |
| `AI.ToolNeeds`, `AI.ToolGrant` | changed: `runs_programs`; `runProgram`, the program folders and the container |
| `AI.PublicAddress` | changed: `nonPublicRanges()`, the deny list of 4.5 drawn from its own table |
| `Mila/Apps/Assistant/Session/Src/Assistant.CommandTool.ixx` | `run_command` |

## 8. Phases

1. **Measure what the design still assumes.** Done 2026-10-27 (`Notebooks/MxcConformance.md` section 7): the
   thread's desktop holds; a denied file is refused, its folder still listed; the streaming surface stops a run in
   40 ms and caps its output; the environment is set by Mila; telemetry stays off; and two findings changed the
   design -- the walk's cost (4.3, decision 7) and the registry (4.9, decision 8).
2. **The library.** Built 2026-10-31 (`Tests/AI/ToolGrant.Programs.cpp`, `Tests/AI/SecondNameWatch.cpp`). The
   watcher of 4.3 measured first, outside the repository. Then section 7's types, the walk and its watcher, and
   the grant's `runProgram`, with tests that each pair an access the grant's
   container refuses with the same access inside it accepted -- a file outside, a name inside of a file outside, a
   file also named in a denied folder, a write to a read-only folder, the network with none granted, a private
   address with the network granted, the screen -- run where the container is available and skipped where it is
   not, saying so. Each of 19 changes that breaks one of those refusals fails a test.
3. **The tool.** `run_command`, the setting, the program folders, the question showing them; the installer carrying
   `mxc_ffi.dll`. Gate: the session drives a build and a `git status` in a chosen folder on Gemma 4 12B; the
   window shows the question with every field of 5.
4. **Later, through the same call:** MCP servers started as processes (`ToolCalling.md` 6), code an agent writes,
   and Linux through bubblewrap once the grant runs there.

## 9. Decisions

All six decided 2026-10-27 (Todd, "Agreed to 1..6"), each as recommended:

1. **v0.21.0's scope.** `ROADMAP.md` *The Desktop Agent* carries the criterion: *with the experimental setting on,
   on Windows 11 26H2, the Mila app runs a command in the chosen folder, asked first, and every phase 2 check
   refuses its access beside the same access inside allowed*; `BACKLOG.md` carries the item. The item's date is the
   release's, so it drains to `Vnext.md` if it does not land, and the criterion with it.
2. **A withheld file is denied for the run, not the run refused** (4.3), and the result names it.
3. **The desktop: the session's thread if phase 1 shows it holds, else a host process per run** (4.4). Phase 1
   showed it holds: the thread.
4. **The shell is Windows PowerShell 5.1:** on every Windows 11, and it starts on Mila's desktop (measured).
5. **The grant keeps its own hard-link rule for v0.21.0** (`ToolCalling.md` 3.4); accepting a file whose names all
   lie inside is filed in `Untriaged.md`.
6. **Microsoft is told after this design**, Todd filing (`Untriaged.md`).

Decided 2026-10-31 (Todd, "7c and 8a", then "Agreed" to the bound and to 9 and 10), after phase 1:

7. **The walk is paid once and the folders are watched** (4.3): a full walk costs 5.5 s on a 358,000-file project,
   so the granted folders are walked before the first run and watched (`ReadDirectoryChangesW`) after it, each run
   re-reading only the files the watcher named. A watcher that overflows has its folder walked again, and a walk
   past a bound refuses the run. The bound, agreed 2026-10-31 after timing (4.3), is the call's own time, never a
   count of files. **Narrowed the same day** (Todd, "A"), when phase 2's tests found NTFS reports no name made
   outside for a file inside: the watcher is relied on only on ReFS, where it was measured to report one; on any
   other file system the folders are walked whole before every run. Not chosen: reading NTFS's change journal
   between runs, whose unprivileged read is unmeasured (a later speed-up); a known gap on NTFS.
9. **The program folders reach only the program** (4.6): `runProgram` gives them to the run, and no file tool
   reads them. (Todd, 2026-10-31.)
10. **A program folder the person can write to is walked and watched as a granted folder is** (4.3); one only an
    administrator can write to, such as under `Program Files`, is not, since no second name can be made inside it
    without that. (Todd, 2026-10-31.) Not chosen: walking only between runs,
   which leaves a name made from outside between runs unseen; a bound alone, which leaves large projects without
   the shell.
8. **The registry is a known gap** (4.9), stated where the setting is beside the hard-link one, and told to
   Microsoft with the other two findings (`Untriaged.md`). The person's saved credentials are refused, `HKCU`
   cannot be written, and the run's own environment is Mila's (4.7). Not chosen: keeping the shell out until MXC
   can refuse it.

## 10. Relationship to other specs

- **`ToolCalling.md`** owns the grant: 3.1 names the fourth need, 3.4 the container as the grant's, and 6 points
  here (2026-10-27).
- **`ToolBroker.md` 6.3** and **`Notebooks/LocalToolService.md` 10.4** first named MXC; `Vnext.md`'s sandbox entry
  holds what phase 4 leaves.
- **`MilaApp.md`** 3.1 (Build & Code's tools) and 9.4 (what the installer carries) change in phase 3.
- **`Notebooks/MxcConformance.md`** holds the measurements this design rests on.
