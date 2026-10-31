# MXC Conformance

What Microsoft eXecution Container (MXC) v1.0.1's policy accepts and refuses, measured 2026-10-26 and 2026-10-27 on
Todd's machine, as evidence for `ContainedPrograms.md`. Each check is one access, first shown to be accepted run
directly, then run under the policy, which must accept or refuse it as declared. Nothing here is in the repository's
code: the programs and scripts are in `D:\Claude\mxc-cpp` (Windows) and `D:\Claude\mxc-linux` (Linux), each with
its logs.

## 1. Setup

- **Windows:** Windows 11 26H2, build 26300.9711 (MXC's process container needs 26300.9550 or later). A C++23
  program loads `mxc_ffi.dll` from the v1.0.1 release and calls `mxc_run_json` with one request (`main.cpp`,
  `build\mxc-run.exe --request <file>`). The `repr(C)` layouts are mirrored from `src/ffi/mxc_ffi/src/lib.rs`;
  MXC publishes no C header (microsoft/mxc#1449).
- **Linux:** WSL 2, Ubuntu 26.04 (`Ubuntu-Dev`, its disk on D:), bubblewrap 0.11.1, `"containment": "bubblewrap"`.
  `libmxc_ffi.so` built from the v1.0.1 tag with Rust 1.93 (MXC's pinned toolchain); the same program through
  `dlopen` (`mxc-run.cpp`).
- **The fixture:** a read-write folder `grant`, a read-only folder `grant-ro`, a denied folder `grant\denied`, and a
  folder `outside` named in no list. Each check runs on a fresh fixture.

## 2. Files

| Access | Windows | Linux |
|---|---|---|
| Read, write inside the read-write folder | accepted | accepted |
| Read, write a file outside every folder | refused | refused (not present) |
| Read the read-only folder; write it | accepted; refused | accepted; refused |
| Read the denied folder | refused | refused |
| A junction, or a symbolic link, inside leading outside: read, write | refused | refused (symbolic link) |
| Make a junction or link inside to an outside file during the run, then read | refused | refused |
| A UNC loopback path, a `\\?\` path, the repository, `/mnt/d` | refused | refused |
| `C:\Windows` / `/etc` | accepted | accepted |
| The user profile / the home folder | refused | lists only the path down to the grant |
| **A file with one name inside the read-write folder and one outside, made before the run: through its inside name, read and write** | **accepted** | **accepted** |
| That file through its outside name, during the run | refused | refused |
| The same with the inside name in the read-only folder: read; write | accepted; refused | accepted; refused |
| **A file named in the read-write folder and in the denied folder: through the read-write name** | **accepted** | **accepted** |
| That file through its denied name | refused | refused |

The policy decides by the name used to open a file. On Windows no permission on the fixture changes before, during
or after a run, or after the host is killed (`icacls`, the whole fixture), and the fixture carries no AppContainer
entry, yet the contained process -- an AppContainer at low integrity, its `TEMP` under a per-run
`Packages\sandbox.{guid}` removed afterwards -- reads and writes it. MXC's documentation does not say how; its issues
name a filter driver, `bfs.sys` (microsoft/mxc#1487), which fits what was measured.

## 3. Processes

| Access | Windows | Linux |
|---|---|---|
| A child process reads outside | refused | refused |
| PowerShell reads outside | refused | -- |
| A process started through WMI, through COM `ShellExecute`, through Task Scheduler, writes outside | refused | -- |
| A child started to outlive the run writes a file after it ends | refused | refused |
| The host killed mid-run: the contained processes | ended at once | ended at once |
| Signal a host process | -- | refused (not visible) |

Launch, a trivial command: 36 to 38 ms on Windows, 8 to 10 ms on Linux.

## 4. Network

| Destination | egress deny | default allow | allow, deny list of IANA's special-purpose ranges |
|---|---|---|---|
| Public HTTPS | refused | accepted | accepted |
| The host's loopback, IPv4 and IPv6 | refused | refused | refused |
| Loopback written as `::ffff:127.0.0.1` | refused | refused | refused |
| The LAN gateway | refused | accepted | refused |
| A name resolved | refused | accepted | accepted |

On Linux egress deny gives the process a network namespace of its own holding only loopback: public HTTPS, the
host's loopback and name resolution are refused. An allow policy is refused at start until `slirp4netns` is
installed, so the deny-list policy was not measured there. Not measured on either: public IPv6.

## 5. The UI

With the UI disabled, a program that loads `user32.dll` does not start (`0xC0000142`): of the build tools, only
`cmd`, `findstr`, `tar`, `ninja`, `nvcc` and `rg` start; `git`, `cmake`, `cl`, `node`, `pwsh`, `powershell`,
`dotnet`, `whoami` and `py` do not. With the UI on, all of them start, `git status` and Node's
`realpathSync.native` included (microsoft/mxc#1464 did not reproduce here).

With the UI on, by `processContainer.ui.isolation` (a native check program, `ui-probe.exe`, which reports only
whether an access succeeded, never what it read):

| Access | container | handles | atoms | desktop |
|---|---|---|---|---|
| **Read the screen (a corner's hash equal to the host's)** | **accepted** | **accepted** | **accepted** | **accepted** |
| See other processes' windows | refused | refused | accepted | accepted |
| Send a message to the taskbar window | refused | refused | accepted | accepted |
| Find a global atom the host added | refused | accepted | refused | accepted |
| Injected keystrokes reaching the host's input (a low-level keyboard hook) | refused | refused | refused | refused |
| Create a desktop; change a system setting; read the clipboard | refused | refused | refused | refused |

Injected input was refused with `injection` false and true alike, though `SendInput` reported both events sent; the
foreground window was at medium integrity, so the refusal may be the integrity check's. Desktop creation, settings
and the clipboard stayed refused with `desktopSystemControl` true, `systemSettings` all and `clipboard` read
(microsoft/mxc#1245).

**On a desktop of its own.** The host program started on a new desktop in `WinSta0` whose DACL grants SYSTEM, the
user, *ALL APPLICATION PACKAGES* and *ALL RESTRICTED APPLICATION PACKAGES*, labelled low integrity
(`S:(ML;;NW;;;LW)`), with the UI on: `git status`, `cmake`, `cl`, `node`, `pwsh` and `powershell` all started; reading
the screen, seeing another window, finding the taskbar, reading the clipboard and injecting input were all refused,
and no injected event reached the host. On a new desktop with the default DACL, nothing that loads `user32.dll`
starts, as microsoft/mxc#1465's comments describe for AppContainer children.

## 6. leastPrivilege, environment, the C library

- `processContainer.leastPrivilege`: `cmd` starts and the file rules hold; `whoami` ("The handle is invalid"), `git`
  ("Access is denied"), `curl` ("error initializing curl library") and PowerShell do not, with the UI on or off. Its
  container profile is one fixed `Packages\cli`, kept across runs.
- The environment: 55 of the host's 110 variables reach the run; a variable set only in the host's environment did
  not.
- `mxc_ffi` exports, besides `mxc_run_json`: `mxc_spawn_json`, `mxc_sandbox_kill`, `mxc_sandbox_try_wait`,
  `mxc_sandbox_wait` and stdout/stderr streams -- a run that is stopped, and output read as it comes. Not exercised.

## 7. Phase 1 of `ContainedPrograms.md`, 2026-10-27

Scripts `policy-conformance-7.ps1` and `-8.ps1`, `walk-time.exe`; the host program gained `--thread-desktop` and
`--spawn`.

- **The calling thread's desktop is enough.** A run started from a thread set (`SetThreadDesktop`) to a new desktop
  with section 5's DACL and label lands on it: the screen read fails, no other window is seen, the taskbar is not
  found, clipboard read and injection are refused (no event reached the host), and `git status`, `cl`, `node` and
  PowerShell start.
- **The streaming surface.** `mxc_spawn_json`, a stream read on its own thread per output, `mxc_sandbox_try_wait`
  polled every 20 ms: a 7-byte output ends in 53 ms; 1,188,895 bytes stream from 30 ms, kept to a 4,096-byte cap
  while the rest is read and dropped, ending in 1,702 ms; `mxc_sandbox_kill` at 2,000 ms ends the run at 2,040 ms with
  no process left. **The request's `timeout` does not end a run that is polled with `try_wait`** -- it is honoured by
  `mxc_sandbox_wait` (streaming.rs) -- so a caller that polls ends it itself (`mxc_sandbox_kill_for_timeout`).
- **`deniedPaths` naming a file**, inside the read-write folder: reading, writing, replacing it by rename and
  deleting it refused; the same for a file whose other name lies outside; the files beside them accepted. A denied
  file is still listed by name. `cmd`'s `dir` is refused in any granted folder, with nothing denied too (it opens the
  volume's root, cf. microsoft/mxc#1461); `for %f in (*)`, PowerShell's `Get-ChildItem` and `git status` list.
- **No folder granted.** Listing `C:\`, `D:\`, `C:\Program Files`, `C:\ProgramData`, `C:\Users`, `C:\Users\Public`:
  refused. Reading a file in `C:\Program Files` by its path: accepted; one in `C:\ProgramData`, and the repository:
  refused. **The registry:** `HKLM` read accepted; under `HKCU`, reading `Environment`, `Software`,
  `Software\Microsoft`, `Software\Classes`, Explorer's `Shell Folders` and `Control Panel\Desktop` accepted,
  `Volatile Environment`, `Console` and `...\CurrentVersion\Run` refused; writing under `HKCU` refused. **Credentials:**
  `cmdkey /list` and listing the Windows Credentials vault refused; `vaultcmd /list` (the vaults' names only)
  accepted. MXC's request has no registry field.
- **The environment.** Without `process.env`, a run gets the person's default user environment -- their user and
  system variables (55 here, `CUDA_PATH`, `OneDrive` and a third-party agent's `HERMES_HOME` among them), not the
  host process's. With `process.env` the list is used as written; MXC refuses to start one without `LOCALAPPDATA`
  ("presence-only"). Nine variables -- `SystemRoot`, `windir`, `ComSpec`, `PATHEXT`, `PATH`, `TEMP`, `TMP`,
  `LOCALAPPDATA` and `PROMPT` as `cmd` adds it -- start `cmd`, `git status`, `cl`, `node` and PowerShell.
- **Telemetry.** MXC's documentation (`README.md`, `docs/telemetry.md`, `docs/development/architecture/telemetry.md`)
  and source: Microsoft's official builds can send optional ETW diagnostic events, source builds cannot; an event is
  sent only when the run's request sets `telemetry.enabled: true` (the SDK logs "not requested for this run"
  otherwise), the Windows user has consented through MXC's own prompt, and `HKLM\SOFTWARE\Policies\Mxc\AllowTelemetry`
  does not block it. Here: consent `undetermined`, policy `unrestricted`, and no TCP connection owned by the host
  process in six samples during a run.
- **The walk** (`FindFirstFileExW`, then each file's link count): `GetFileInformationByName` and opening each file
  cost the same. 7,484 files in 1,015 folders (the Mila app's `node_modules`): 113 to 119 ms. The repository with its
  build directories, 357,961 files in 35,479 folders: 5.5 s; 2 files with more than one name, every name inside.
- **The watcher** (2026-10-31, `watch-probe.cpp`; `ReadDirectoryChangesW` on the inside folder, whole tree, D:,
  which is ReFS, a Dev Drive -- first written down here as NTFS, wrongly): a name made inside for a file outside is reported as an added name. A name made outside for a file inside
  is reported as a change to the inside file, under `FILE_NOTIFY_CHANGE_ATTRIBUTES` and under no other filter. Not
  reported: an outside name removed, and a write through a file's outside name. 20,000 files written into the folder
  in 960 ms: 40,001 events read, none lost, with a 64 KB buffer. `npm ci` of the Mila app's packages into a watched
  folder (7,410 files, 5.8 s): 35,866 events, none lost. A clean rebuild of every Mila target (1,801 steps, 350 s)
  with the whole repository watched: 18,123 events and no overflow with a 64 KB buffer; a 4 KB buffer beside it
  overflowed 19 times.
- **The watcher on NTFS** (2026-10-31, `Tests/AI/SecondNameWatch.cpp` under the system's temporary folder on C:,
  NTFS): a name made outside for a file inside is **not reported under any filter**; the same test on D: (ReFS)
  passes. A name made inside, a file's other name removed, and a folder moved in are reported on both. NTFS's change
  journal records a link added or removed (`USN_REASON_HARD_LINK_CHANGE`, `USN_RECORD_V2` docs); reading it without
  an administrator is not measured.

## 8. Others using MXC

Read 2026-10-27, from each project's source:

- **Codex** (`openai/codex`, `codex-rs/mxc-sandbox/src/policy.rs`) sets the UI on at isolation `desktop`, "PowerShell
  needs Win32k and desktop handles during DLL startup"; its README names clipboard, input injection and system
  control as restricted, and not the screen. It lists hard links among the cases its smoke suite must cover.
- **NVIDIA OpenShell** (`crates/openshell-driver-mxc/src/policy_map/map.rs`) sets the UI disabled.
- MXC's issues hold none on reading the screen or on files with more than one name (searched 2026-10-27). Both are
  for Microsoft's security response centre, which `SECURITY.md` names in place of public issues (`Untriaged.md`).
