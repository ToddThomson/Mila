# Contained Programs

How a tool runs a program held to its grant, through Microsoft eXecution Container (MXC), on Windows and on macOS,
and the first tool that does: Mila Assistant's `run_command`.

**Status:** agreed 2026-10-27 (Todd, decisions 1 to 6 of section 9); phase 1 measured the same day; decisions 7
to 11 agreed 2026-10-31, and phases 2 and 3, the library and the tool, built that day on Windows; phase 4 not begun.
**The macOS half is missing, so phases 2 and 3 are not done** (decision 12, Todd, 2026-11-17): section 11 proposes it,
not yet measured. Decided before it (Todd, 2026-10-27): MXC goes into
tool calling for v0.21.0 as an experiment, behind a switch that is off by default, as long as the grant
(`ToolCalling.md` 3.4) remains v0.21.0's safe tool calling; the design works around the two accesses MXC's policy
accepts that no grant gives; a file that gains a second name from outside while a run is going is accepted as a
known gap. Evidence: `Notebooks/MxcConformance.md`. Decisions in section 9. **No longer an experiment** (Todd,
2026-11-26, decision 16): v0.21.0 is fully agentic, so running commands is on by default and is a feature of Windows,
macOS and Linux alike.

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
            MxcProgramContainer                macOS: libmxc_ffi.dylib, MXC's Seatbelt backend (section 11, proposed)
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
left alone -- as built; decision 15 replaces this with denying every such file, on both platforms -- and any other is
put in `deniedPaths` for that run, by its name inside, and the result names it so the model
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
  installing Mila Assistant's npm packages, or rebuilding every Mila target with the whole repository watched
  (18,123 events in 350 s); a 4 KB buffer overflowed 19 times in that build. So a walk after the first is rare.
- **The bound** (Todd, 2026-10-31): no count of files. A walk, the first or one after an overflow,
  is part of the run and counts against the call's own time; a walk that does not finish within it refuses the run,
  saying the folder is too large to check in the time given.
- **macOS and Linux** have no call that lists a file's names, so a file with more than one name is denied there
  whatever its other names; for macOS, and whether Windows keeps the rule above, see section 11.3. Linux is v0.21.0's
  (decision 16), through MXC's bubblewrap backend, and stands on the grant's file checks there.

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

As built; decision 14 puts a proxy of Mila's own in front of this on both platforms, and the list stays beneath it
on Windows (11.4). The grant's network is public addresses only (3.4). MXC filters by address, so the grant's rule
becomes a deny list:
every IANA special-purpose range, IPv4 and IPv6, and multicast, generated from `PublicAddress`'s own table so the
two cannot drift. The IPv6 forms that carry an IPv4 address -- IPv4-mapped, NAT64, 6to4 -- are denied whole, which
is stricter than the grant, which decides by the IPv4 address inside. Measured on Windows: public HTTPS accepted; the
LAN gateway, loopback and loopback written as `::ffff:127.0.0.1` refused. Names still resolve under that policy,
though the LAN's resolver is in a denied range; MXC does not document how, and with no network they do not.
Loopback is refused by MXC's own host rule as well, so it cannot show that the list is sent; the LAN gateway is
refused by the list alone, and its test is the one that fails without it.

### 4.6 Program folders

A command runs programs that live outside the project: Git, CMake, Visual Studio's compiler, Node. The application
names those folders, read-only, as part of the grant it gives the session (`Session::withGrant`), with the
variables those tools need (`ProgramFolders`: a compiler's `INCLUDE`, say), and the person is shown the folders.
Neither is ever the model's to choose. Measured in phase 1: with no folder granted, a file under
`C:\Program Files` is read by its path while the folder cannot be listed, so granting the program folders is what
lets the tools list and search their own; `C:\ProgramData`, `C:\Users` and other drives are refused.

### 4.7 The environment

A run is given the environment Mila writes (`process.env`), never MXC's default, which is the person's own user
and system variables -- whatever a person keeps there, a token included. The list:

- `SystemRoot`, `windir`, `ComSpec`, `PATHEXT`, and `PATH`: System32, Windows PowerShell and the program folders.
- `LOCALAPPDATA`, the person's own. MXC refuses to start without it, and puts it, `TEMP` and `TMP` in the run's own
  AppContainer profile beneath it, which Windows makes only under the person's own and removes after the run.
  Pointing it into the project, as first built, left the temporary folder missing (`MxcConformance.md` 7.1).
- `USERPROFILE` and `HOME` as `.mila\Home`, and `APPDATA` as `.mila\Roaming`, in the project: the tools' settings
  and caches stay in the project, never in the person's profile.
- `SystemDrive`, `ProgramFiles`, `ProgramFiles(x86)`, `CommonProgramFiles`, `ProgramData`, `ALLUSERSPROFILE`:
  names of the system's folders, which tools read to find their own; naming one grants nothing.
- Then the application's own variables (`ProgramFolders`, 4.6), refused at the grant when one is a name above.

The project is the first folder granted to write; a grant with none refuses to run a program. Measured: with
these and Mila Assistant's variables, `git status` and `git commit`, a CMake + Ninja + `cl` build, `npm run build`
and `dotnet build` finish (`MxcConformance.md` 7.1).

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
  runs and a machine without it offers no program-running tool. Mila Assistant's installer carries the file
  (MIT, about 8 MB at v1.1.0) beside `mila-session`. **Moved to v1.1.0 on 2026-11-17:** the eleven functions Mila
  calls and the error detail it mirrors are unchanged from v1.0.1 (the FFI crate's source at both tags compared);
  v1.1.0 adds `mxc_spawn_extended_json`, `mxc_register_runtime_directory` and two state-aware calls, none used.
- **The version is pinned.** MXC publishes no C header (microsoft/mxc#1449), so Mila mirrors the `repr(C)` layouts of
  one version, checks `mxc_version()` on load, and refuses any other.
- **Availability is asked, not assumed:** `mxc_platform_support_json` at load; on a Windows build older than
  26300.9550 the container is absent and so is the tool.
- **Telemetry stays off.** Microsoft's builds of MXC can send diagnostic events; one is sent only when the run's
  request enables it, the person has consented through MXC's own prompt, and administrative policy allows it
  (MXC's `docs/telemetry.md`). Mila writes `telemetry.enabled: false` in every request and never shows MXC's prompt,
  so another application's consent changes nothing for Mila's runs. Phase 1: consent `undetermined`, and no
  connection owned by the host during a run.
- **From v1.1.0 that switch no longer holds** (MXC's `docs/telemetry.md` and `docs/development/architecture/telemetry.md`
  at v1.1.0, read 2026-11-17). Microsoft's builds send four Required events for every request contract, `1.0.0`
  included: `MXC.ExecutionStart`, `MXC.ExecutionEnd`, `MXC.Error` and `MXC.ProcessExited`, carrying the backend, the
  outcome, the exit code, the duration, a failure category, a random per-run session id and MXC's version, and by
  MXC's account no command line, path, environment value or output. `telemetry.enabled: false`, consent and
  administrative policy do not stop them, and Windows' own diagnostic-data setting does not govern them. They are
  routed only by a build that carries the signed metadata of `mxc_telemetry.dll`; a release built from source with
  `MXC_TELEMETRY_COMPONENT_PATH` empty is unrouted, its events local ETW that nothing sends (unmeasured). macOS has no
  MXC telemetry. **Decided (Todd, 2026-11-17, decision 13):** development may run Microsoft's v1.1.0 binary; what
  Mila ships is an unrouted build of its own, confirmed unrouted by a local trace before it ships.
- **Windows versions.** MXC's own table (`docs/backends/process-container/os-version-support.md`, v1.1.0) lists
  process isolation from 26100.9278 on 24H2 and 26200.9278 on 25H2, so "26H2 or later" in section 6 may be narrower
  than MXC; Mila asks MXC at load either way. Unmeasured: no 24H2 or 25H2 machine has been tried. From v1.1.0
  ProcessContainer runs only on Windows' native PSEC and refuses a request PSEC cannot enforce before launch; the
  AppContainer tier and the host-ACL fallback are gone.

### 4.11 The switch

- **The build:** a CMake option, off by default, selects `MxcProgramContainer`'s module or a module whose container
  is always absent -- two candidate files for one module, as the grant's file system has (no `#ifdef` in a module).
  Off, nothing of MXC is built or shipped.
- **The app:** a setting, on by default since decision 16 (off by default and labelled experimental until then). Off,
  the shell tool is not registered. On where the container cannot run, the setting says why and Mila tells the person
  it cannot run commands on this computer.

## 5. The first tool: running a command in Build & Code

- **`run_command`**, Mila Assistant's own, not one of the tools Mila::Agents provides (`ToolCalling.md` 5): arguments `command`, an optional time, and an optional
  `network`, false unless the model sets it; needs: reads and writes files, runs programs, uses the network. The
  conversation's grant holds the network, and the tool narrows it away for every call that does not ask for it, so
  a call never reaches more than its grant (4.1). With the network, a command reaches public addresses only (4.5).
- **Always asked.** No default rule allows it, and the question shows the command whole, the folder, the program
  folders and whether the network. A command that needs the network is asked as one, as Codex asks before a command
  that needs it (decision 11).
- The command runs under Windows PowerShell 5.1, present on every Windows 11 (decision 4), given whole as
  a script the tool writes through its grant to `.mila\command.ps1` and runs with `-File`, so no quoting of the
  model's text changes the command line and errors come back as text, with execution policy bypassed for that
  process alone so `npm.ps1` runs. `cmd`'s `dir` is refused in a granted folder (phase 1), so the prompt names
  PowerShell's own commands.
- **What Mila Assistant gives a command** (`Assistant.ProgramSetup`, found once when the setting is turned on): the
  folders of Git, CMake, Ninja, Node and .NET on the person's `PATH`; Visual Studio's x64 setup, run once on the host,
  for `INCLUDE`, `LIB`, `LIBPATH`, the folders they name and the folders under Visual Studio and the Windows SDK it adds
  to `PATH`; `NODE_OPTIONS=--preserve-symlinks --preserve-symlinks-main`, since Node resolves a script's real path
  from the drive's root and the container refuses that root; `_CL_=/Z7`, so a project asking for `/Zi`, whose
  separate PDB process cannot be reached, embeds its debug information instead; .NET's home in the project
  (`DOTNET_CLI_HOME`), its first-run work and telemetry off, and its Git query off
  (`EnableSourceControlManagerQueries=false`), since that query reads the person's own `.gitconfig` and is refused.
  None of it is the model's to know: each was measured (`MxcConformance.md` 7.1) and is set by variable. The
  person's Git name and email, and nothing else of their Git settings, go in the project's `.mila\Home\.gitconfig`,
  and `.mila\.gitignore` keeps the tools' home and the script out of the project's history.
- The result the model reads: the exit code, the output as cut, and any files withheld (4.3).
- **What the model is told** (2026-11-26, after Todd's walk, where a plan wrote a `Makefile` for `g++` and said it
  could not build): while `run_command` is offered, which build tools a command can use here -- those of the program
  folders above, by name -- and none other; that Windows PowerShell 5.1 has no `&&`; and that a build goes outside the
  sources and a first commit comes with a `.gitignore` for it. The chosen folder's result says it can be written as well
  as read, since "to read" alone was taken as read-only. Before the chat has a folder, the model is told that commands will come with one
  when the setting is on, with it off, that the person turns them on in Settings, and with it on where the container
  cannot run, that this computer cannot run them, so a plan
  made before the folder neither gives up nor promises.

## 6. What the person is told

The setting says what it does and what it does not: commands run in a Windows container limited to the project
folder and the program folders shown, on a desktop of their own, with no network unless a command asks for it and
the person allows it, and then public addresses only, where an allowed command can send what it reads; on Windows, on
Windows 11 26H2 or later. It never calls the tool sandboxed. The known gaps -- a file given a second name during a
run (4.3), and the registry under `HKCU` read (4.9) -- are stated where the setting is.

## 7. Modules

One type each, under `Lib/Src/Agents/` unless named:

| Module | Type |
|---|---|
| `Agents.ProgramRequest` | `ProgramRequest` |
| `Agents.ProgramResult` | `ProgramResult` |
| `Agents.ProgramFolders` | `ProgramFolders`, the application's program folders and variables (4.6), and the names it may not give |
| `Agents.ContainedRun` | `ContainedRun`, what the grant hands its container: the command and every folder by its final path |
| `Agents.ProgramContainer` | `ProgramContainer`, the interface |
| `Agents.MxcProgramContainer` | the MXC implementation, and its always-absent twin, two candidate files |
| `Agents.ToolDesktop` | the desktop of 4.4, created once and closed with the container |
| `Agents.SecondNameWatch` | the walk and watcher of 4.3 for one granted folder: the files with a name elsewhere |
| `Agents.ToolNeeds`, `Agents.ToolGrant` | changed: `runs_programs`; `runProgram`, the program folders and the container |
| `Networking.PublicAddress` | changed: `nonPublicRanges()`, the deny list of 4.5 drawn from its own table (moved from `Agents`, `Network/Network.md` 4.2) |
| `Apps/Assistant/Session/Src/Tools/Assistant.CommandTool.ixx` | `run_command` |
| `Apps/Assistant/Session/Src/Tools/Assistant.ProgramSetup.ixx` | the build tools' folders and variables of section 5 |

## 8. Phases

1. **Measure what the design still assumes.** Done 2026-10-27 (`Notebooks/MxcConformance.md` section 7): the
   thread's desktop holds; a denied file is refused, its folder still listed; the streaming surface stops a run in
   40 ms and caps its output; the environment is set by Mila; telemetry stays off; and two findings changed the
   design -- the walk's cost (4.3, decision 7) and the registry (4.9, decision 8).
2. **The library.** Built 2026-10-31 (`Lib/Tests/Agents/ToolGrant.Programs.cpp`, `Lib/Tests/Agents/SecondNameWatch.cpp`). The
   watcher of 4.3 measured first, outside the repository. Then section 7's types, the walk and its watcher, and
   the grant's `runProgram`, with tests that each pair an access the grant's
   container refuses with the same access inside it accepted -- a file outside, a name inside of a file outside, a
   file also named in a denied folder, a write to a read-only folder, the network with none granted, a private
   address with the network granted, the screen -- run where the container is available and skipped where it is
   not, saying so. Each of 19 changes that breaks one of those refusals fails a test.
3. **The tool.** `run_command`, the setting, the program folders, the question showing them; the installer carrying
   `mxc_ffi.dll`. Gate: the session drives a build and a `git status` in a chosen folder on Gemma 4 12B; the
   window shows the question with every field of 5. Built 2026-10-31, after measuring the builds the environment of
   4.7 as first built did not finish (`MxcConformance.md` 7.1): on Gemma 4 12B Q4_0 the session ran `git status`, a
   CMake + Ninja build and the program it built, and `npm run build`, and an install asked for the internet and,
   declined, installed nothing; the window showed the question with the command, the folder, the program folders
   and the internet, offering Allow once and Deny alone.
4. **Later, through the same call:** MCP servers started as processes (`ToolCalling.md` 6), code an agent writes,
   and Linux through bubblewrap once the grant runs there.
5. **macOS** (section 11), v0.21.0 work with phases 2 and 3, which are not done without it. First the conformance
   checks of `Notebooks/MxcConformance.md` section 7 run on the Mac against MXC's Seatbelt backend at v1.1.0; then
   the Mac candidate file, phase 2's tests passing on both machines, and phase 3's gate on both. It stands on the
   grant's macOS file checks (`ToolCalling.md` 3.4, BACKLOG *Mila on Apple Silicon*), which come first.

## 9. Decisions

All six decided 2026-10-27 (Todd, "Agreed to 1..6"), each as recommended:

1. **v0.21.0's scope.** `ROADMAP.md` *The Desktop Agent* carries the criterion: *with the experimental setting on,
   on Windows 11 26H2, Mila Assistant runs a command in the chosen folder, asked first, and every phase 2 check
   refuses its access beside the same access inside allowed*; `BACKLOG.md` carries the item. The item's date is the
   release's, so it drains to `Vnext.md` if it does not land, and the criterion with it. Widened to macOS 15 or
   later by decision 12.
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
   count of files. Not chosen: walking only between runs, which leaves a name made from outside between runs unseen;
   a bound alone, which leaves large projects without the shell. **Narrowed the same day** (Todd, "A"), when phase 2's tests found NTFS reports no name made
   outside for a file inside: the watcher is relied on only on ReFS, where it was measured to report one; on any
   other file system the folders are walked whole before every run. Not chosen: reading NTFS's change journal
   between runs, whose unprivileged read is unmeasured (a later speed-up); a known gap on NTFS.
8. **The registry is a known gap** (4.9), stated where the setting is beside the hard-link one, and told to
   Microsoft with the other two findings (`Untriaged.md`). The person's saved credentials are refused, `HKCU`
   cannot be written, and the run's own environment is Mila's (4.7). Not chosen: keeping the shell out until MXC
   can refuse it.
9. **The program folders reach only the program** (4.6): `runProgram` gives them to the run, and no file tool
   reads them. (Todd, 2026-10-31.)
10. **A program folder the person can write to is walked and watched as a granted folder is** (4.3); one only an
    administrator can write to, such as under `Program Files`, is not, since no second name can be made inside it
    without that. (Todd, 2026-10-31.)

Decided 2026-10-31 (Todd, "off by default", then "Yes to your 2 step plan"):

11. **A command has no network unless it asks and the person allows it** (section 5). Codex starts commands with
    the network off and asks before a command that needs it; Claude Code's sandbox starts with no domain allowed and
    asks for each new one (their documentation, read 2026-10-31). Phase 3 asks per call, by the `network` argument.
    A list of hosts allowed per project, asked for each new host, follows it: MXC's rules match addresses only and
    no name (its networking document), so that list needs MXC's proxy mode and a proxy of Mila's own, measured
    first (`Untriaged.md`). Not chosen: the network on for every command, which lets any allowed command send what
    it reads.

Decided 2026-11-17 (Todd):

12. **Running a program is a feature of Windows and macOS alike** ("win and Mac have the same features. It should be
    that way by design"). The design names each platform's mechanism, and where one lacks a primitive the shared
    design changes rather than one side going without (section 11). Phases 2 and 3 are done only when both halves
    are.
13. **Telemetry: development may use Microsoft's v1.1.0 binary; Mila ships an unrouted build of its own** (4.10).

Decided 2026-11-17 by Claude, Todd having left them to its call and to review later ("7,8,9 your call - I'll review
later"):

14. **A command's network goes through a proxy of Mila's own on both platforms** (11.4), deciding by `PublicAddress`;
    Windows keeps the deny list of 4.5 beneath it. Not chosen: no network for commands on the Mac, which leaves the
    Mac short; the deny list alone on Windows and the proxy on the Mac, two rules for one feature.
15. **A file with more than one name is denied for the run on both platforms, whatever its other names** (11.3), as
    the grant's file checks refuse one. Not chosen: Windows' rule of 4.3 beside a stricter Mac rule, two behaviours
    for one feature, and one that no grant gives in the first place.

Decided 2026-11-26 (Todd: "No, commands are no longer experimental. 0.21 will be fully agentic", then "Agreed to all"):

16. **Running commands is not an experiment: on by default, on Windows, macOS and Linux.** The setting is on unless
    the person turns it off, and is not labelled experimental; the known gaps of 4.3 and 4.9 are stated where it is as
    facts. Linux joins decision 12, through MXC's bubblewrap backend once the grant's file checks run there, and moves
    from `Issues/Vnext.md` into v0.21.0. What Mila ships carries an unrouted MXC build of its own (decision 13), and
    the release build carries the container (4.11's CMake option), which the maintainer's own build has off today.

## 10. Relationship to other specs

- **`ToolCalling.md`** owns the grant: 3.1 names the fourth need, 3.4 the container as the grant's, and 6 points
  here (2026-10-27).
- **`ToolBroker.md` 6.3** and **`Notebooks/LocalToolService.md` 10.4** first named MXC; `Vnext.md`'s sandbox entry
  holds what phase 4 leaves.
- **`MilaAssistant.md`** 3.1 (Build & Code's tools) and 9.4 (what the installer carries) changed in phase 3.
- **`Notebooks/MxcConformance.md`** holds the measurements this design rests on.

## 11. macOS (proposed 2026-11-17, unmeasured)

The same interface, the same grant and the same switch, through MXC's Seatbelt backend: MXC turns the request into
a profile for Apple's kernel sandbox and applies it between `fork()` and `exec()`, the sandbox lasting exactly as long
as the process tree (MXC's `docs/backends/seatbelt/seatbelt-backend.md` at v1.1.0, read 2026-11-17; every statement
about Seatbelt below is from it and waits on phase 5's checks). It needs macOS 15 or later. Seatbelt is not new in
v1.1.0, but v1.1.0 fixes its symlink and parent-traversal enforcement (microsoft/mxc#1531), so the Mac starts there.

### 11.1 The library and the switch

- `Agents.MxcProgramContainer` gains a third candidate file, macOS's, beside Windows' and the always-absent one;
  `MILA_ENABLE_MXC` selects it on Apple as on Windows. `libmxc_ffi.dylib` is loaded at run time from the
  application's bundle, the same `repr(C)` functions as Windows' (they are one Rust crate), the version pinned the same
  way. MXC's v1.1.0 release carries `arm64/libmxc_ffi.dylib` (about 5.6 MB) beside the Windows DLL; MXC sends no
  telemetry on macOS, so the decision of 4.10 does not reach it.
- The app's setting is the same setting, on by default (decision 16), with the same wording but the platform's.

### 11.2 From the grant to Seatbelt

| Grant | MXC request on macOS |
|---|---|
| Folders granted read-write, read-only; program folders | `readwritePaths`, `readonlyPaths`, as on Windows; MXC resolves symlinks before the profile is made and refuses a dangling one |
| Files with more than one name | `deniedPaths` (11.3) |
| No network | `egress.default: deny`; `ingress.default: deny`, `hostLoopback: deny` |
| The network | not by address: Seatbelt has no address or range rule and refuses one (11.4) |
| -- | `containment: seatbelt`; no `guiAccess`, so the UI, the clipboard and injected input are refused by the kernel, which takes the place of 4.4's desktop |
| -- | `process.env`: Mila's, as 4.7 (11.5); Seatbelt never inherits the host's |

Every sandbox also reads a baseline MXC grants itself: `/bin`, `/sbin`, `/usr/bin`, `/usr/sbin`, `/usr/lib`,
`/usr/libexec`, `/usr/share`, `/System`, `/Library`, parts of `/private/etc` and `/private/var/db`, and the active
developer directory; never the person's own `~/Library`. A read-write folder also lets a program connect to a socket
already listening inside it, which is a reason the project stays the only folder granted to write.

### 11.3 Files with more than one name

macOS gives a file's link count and no call listing its other names, so a file in a granted folder with more than one
name is denied for the run by its name inside, as Linux would be (4.3); a watcher keeps the walk current as 4.3's does,
by the platform's own mechanism, measured first. **Both platforms take that rule** (decision 15): on Windows too, a
file with more than one name is denied for the run whatever its other names, which is the rule the grant's own file
checks already keep (`ToolCalling.md` 3.4). What it costs Windows is only a file whose every name lies inside the
granted folders; pnpm's store and a `git clone` from elsewhere on the drive put a name outside, and so were denied
already. Accepting a file whose names all lie inside stays filed for both platforms (decision 5, `Untriaged.md`).

### 11.4 The network

Seatbelt's outbound network is on or off, with a loopback proxy as the one exception it enforces: with
`egress.default: deny` and `runtimeConfig.networkProxy` naming a loopback port, the kernel lets the sandbox reach that
port and nothing else; whether a program uses the proxy is its own affair, and one that does not has no network. So
"public addresses only" (4.5) cannot come from Seatbelt. **On both platforms a command's network goes through a
proxy of Mila's own** (decision 14), which opens each connection through the grant and so decides by `PublicAddress`
as the grant's own transport does. The container lets the command reach the proxy's loopback port and nothing else:
on macOS by `egress.default: deny` with `runtimeConfig.networkProxy`, on Windows by MXC's proxy mode, whose
enforcement is measured first. Windows keeps 4.5's deny list beneath it as a second layer, which Seatbelt has no
counterpart for; the rule a command meets is the proxy's on both. It is also the base the per-project host list of
decision 11 needs. A program that ignores the proxy has no network, on either platform. Until the proxy exists, a Mac
command has no network, which is a gap against Windows to close in v0.21.0, not the Mac's design.

### 11.5 The environment, the shell and the program folders

- Seatbelt starts from an empty environment. Mila writes it as 4.7 does: `PATH` of `/usr/bin:/bin:/usr/sbin:/sbin`
  and the program folders; `HOME` as `.mila/Home` in the project; a temporary folder in the project or MXC's own, to be
  measured as `TEMP` was on Windows (`MxcConformance.md` 7.1).
- The shell is `/bin/zsh`, present on every supported macOS, given the command as a script the tool writes through
  its grant to `.mila/command.sh`, as PowerShell is given `.mila\command.ps1`.
- `Assistant.ProgramSetup` gains macOS's candidate: Git, CMake, Ninja, Node and .NET where the person's `PATH` finds
  them (Homebrew's prefix included), and Xcode's tools through MXC's developer-directory grant.

### 11.6 What has no counterpart

The registry gap (4.9) has none on macOS; whether the person's defaults under `~/Library/Preferences` are refused is
one of phase 5's checks. MXC sends no telemetry on macOS (4.10).
