<#
.SYNOPSIS
    Build every Mila artifact from the current tree. Builds only -- publishes nothing.

.DESCRIPTION
    RELEASING.md step 1 builds the wheels and step 9 builds the container images. Without this
    script the first time anyone learns either is broken is DURING a release, with a tag already
    pushed and the tree closed to unrelated commits. This runs the same builds from a working tree,
    ahead of time, so a break is found while it is still cheap.

    IT PUBLISHES NOTHING, and that is structural rather than a flag. Phase 4 calls
    build-runtime-image.sh, which only ever writes a local tag; publish-image.sh -- the script that
    can reach Docker Hub -- refuses to run outside a pushed tag with a clean tree, so it cannot be
    invoked from here even by mistake. Nothing touches PyPI, git, or GitHub Pages.

    PHASES ARE STRICTLY SERIAL, and phases 2 and 3-4 especially so. A native MSVC build and a
    Docker build contend for the same machine through WSL2, and running them together has cost
    real time twice. There is deliberately no -Parallel.

    It never configures a build directory it does not own. Phase 1 builds the already-configured
    out/build/x64-claude-verify and never reconfigures it -- configuring wipes a cache someone
    else may be relying on.

.PARAMETER Phase
    Run only these phases (e.g. -Phase 3,4 to resume after a failure). Default: all.

.PARAMETER SkipPhase
    Run everything except these.

.PARAMETER DryRun
    Print the plan, the estimates and the preflight findings, then stop. Touches nothing, reads no
    network, starts no build. Safe to run at any time, including during other work.

.PARAMETER LogDir
    Where per-phase logs go. Default: out/build-artifacts/<timestamp>.

.PARAMETER MinimumFreeGb
    Refuse to start below this much free space on the repository's drive. A two-hour run that dies
    on ENOSPC at minute 90 is the expensive failure this guards.

.EXAMPLE
    scripts/build-artifacts.ps1 -DryRun
    scripts/build-artifacts.ps1
    scripts/build-artifacts.ps1 -Phase 3,4
#>
[CmdletBinding()]
param(
    [int[]] $Phase,
    [int[]] $SkipPhase,
    [switch] $DryRun,
    [string] $LogDir,
    [int]    $MinimumFreeGb = 60
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Path (Split-Path -Path $MyInvocation.MyCommand.Path -Parent) -Parent

# The build directory this script owns. Phase 1 BUILDS it and never configures it: a configure
# wipes the CMake cache, and this tree is shared with whoever else is using the checkout.
$verifyBuildDir = Join-Path $repo "out\build\x64-claude-verify"

# Git Bash, by full path. See the preflight note: bare `bash` is the WSL launcher here.
$gitBash = "C:\Program Files\Git\bin\bash.exe"

# Captured once so the inventory can tell what THIS run produced from what was already on disk.
$runStart = Get-Date
$runStamp = $runStart.ToString( "yyyyMMdd-HHmmss" )

if (-not $LogDir) {
    $LogDir = Join-Path $repo "out\build-artifacts\$runStamp"
}

# ---------------------------------------------------------------------------
# Phase table. One source of truth: -DryRun renders this, the runner walks it.
# `Kind` drives the preflight requirements and the serialization note, nothing else.
# `Estimate` is a COLD guess until a real run replaces it -- see the summary this
# script writes at the end, which reports what each phase actually took.
# ---------------------------------------------------------------------------
$phases = @(
    # EVERY ESTIMATE BELOW IS MEASURED AND WARM -- one full sweep, 2026-09-09, ~48 min to the
    # phase 5 failure. Warm matters: ccache, the BuildKit cache mounts and CPM_SOURCE_CACHE were
    # all populated, and phase 2's --fresh clears the CMake cache without clearing ninja's
    # objects, so it rebuilt 11 edges rather than the tree. A COLD run is still unmeasured and
    # will be substantially longer -- do not quote these to justify a deadline.
    @{ Id = 1; Name = "Library and test suite"; Kind = "native"; Estimate = "~10 min warm"
       What = "cmake --build x64-claude-verify, then ctest. Fail-fast gate for everything below." }

    @{ Id = 2; Name = "Windows wheels (3.12, 3.13)"; Kind = "native"; Estimate = "~4 min warm"
       What = "scripts/pypi/build-wheel-windows.ps1 -> out/wheel/*win_amd64.whl" }

    @{ Id = 3; Name = "Linux wheels (3.12, 3.13)"; Kind = "docker"; Estimate = "~12 min warm"
       What = "docker compose -f Docker/docker-compose.wheel.yml -> out/wheel/*manylinux*.whl" }

    # The longest phase, and the one least helped by a warm cache: MILA_CLEAN_BUILD=1 forces the
    # module graph to recompile, which is the point (it matches what publish-image.sh does).
    @{ Id = 4; Name = "Container images (runtime, devel)"; Kind = "docker"; Estimate = "~22 min"
       What = "scripts/dockerhub/build-runtime-image.sh -> mila-llm:local-{runtime,devel}" }

    @{ Id = 5; Name = "Website (Hugo + Doxygen)"; Kind = "native"; Estimate = "~1 min"
       What = "Doxygen WARN_AS_ERROR + hugo build. Catches what silently blocks publish-site.yml." }

    @{ Id = 6; Name = "Inventory"; Kind = "none"; Estimate = "seconds"
       What = "What was produced, with sizes and versions." }
)

function Test-PhaseSelected( [int] $id )
{
    if ($Phase -and ($Phase -notcontains $id)) { return $false }
    if ($SkipPhase -and ($SkipPhase -contains $id)) { return $false }

    return $true
}

$selected = @($phases | Where-Object { Test-PhaseSelected $_.Id })

function Write-Head( [string] $text )
{
    Write-Host ""
    Write-Host "=== $text " -ForegroundColor Cyan -NoNewline
    Write-Host ("=" * [Math]::Max( 4, 74 - $text.Length )) -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# Preflight. Every check here is something that has actually cost a run, and each one
# reports rather than assumes -- a tool that is present but the wrong version is the
# case that produces a confident, wrong result.
# ---------------------------------------------------------------------------
function Invoke-Preflight
{
    Write-Head "Preflight"

    $findings = [System.Collections.Generic.List[string]]::new()
    $fatal = [System.Collections.Generic.List[string]]::new()

    $version = (Get-Content (Join-Path $repo "Version.txt") -Raw).Trim()
    Write-Host "  Repository      $repo"
    Write-Host "  Version.txt     $version"

    $gitStatus = & git -C $repo status --porcelain 2>$null

    if ($gitStatus) {
        $findings.Add( "working tree is dirty ($((@($gitStatus)).Count) files) -- artifacts will not correspond to any commit" )
    } else {
        $sha = (& git -C $repo rev-parse --short HEAD 2>$null)
        Write-Host "  Commit          $sha (clean)"
    }

    $needNative = @($selected | Where-Object { $_.Kind -eq "native" }).Count -gt 0
    $needDocker = @($selected | Where-Object { $_.Kind -eq "docker" }).Count -gt 0

    if ($needNative) {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

        if (Test-Path $vswhere) {
            $vsPath = & $vswhere -latest -products * -property installationPath
            Write-Host "  Visual Studio   $vsPath"
        } else {
            $fatal.Add( "vswhere.exe not found; the native phases cannot enter a VS developer shell" )
        }

        # The wheel build declares its own toolkit (see build-wheel-windows.ps1). Report what is
        # ambient anyway: a mismatch is not an error, but it IS the thing that makes a local
        # result differ from the one the release will produce.
        $declared = Select-String -Path (Join-Path $repo "scripts\pypi\build-wheel-windows.ps1") `
                                  -Pattern '^\$cudaVersion = if .* else \{ "([\d.]+)" \}' |
                    ForEach-Object { $_.Matches.Groups[1].Value } | Select-Object -First 1

        Write-Host "  CUDA declared   $declared (wheel build pins this)"
        Write-Host "  CUDA ambient    $(if ($env:CUDA_PATH) { Split-Path $env:CUDA_PATH -Leaf } else { '<unset>' })"

        foreach ($py in @("3.12", "3.13")) {
            $found = (& py "-$py" -c "import sys; print(sys.executable)" 2>$null)

            if ($found) { Write-Host "  Python $py      $found" }
            else { $fatal.Add( "Python $py not found; the Windows wheel matrix needs both interpreters" ) }
        }
    }

    if (Test-PhaseSelected 4) {
        # GIT BASH SPECIFICALLY, never bare `bash`. On this machine bare bash resolves to
        # C:\Windows\System32\bash.exe -- the WSL launcher -- where D:/Repos/Mila does not exist
        # and the phase dies on `cd`. Git Bash maps drive letters (/d/Repos/Mila) and, importantly,
        # rewrites the POSIX repo path back into the Windows form docker needs for the build
        # context. Checked here because the failure is otherwise found 90 minutes into a run.
        if (Test-Path $script:gitBash) {
            Write-Host "  Git Bash        $script:gitBash"
        } else {
            $fatal.Add( "Git Bash not found at $script:gitBash; phase 4 cannot run (bare 'bash' is WSL and cannot see the repo path)" )
        }
    }

    if ($needDocker) {
        $serverVersion = (& docker version --format '{{.Server.Version}}' 2>$null)

        if ($serverVersion) {
            Write-Host "  Docker          $serverVersion"

            $df = & docker system df --format '{{.Type}} {{.Size}} {{.Reclaimable}}' 2>$null
            foreach ($line in $df) { Write-Host "                  $line" }
        } else {
            $fatal.Add( "Docker is not responding; phases 3 and 4 need it" )
        }
    }

    if (Test-PhaseSelected 5) {
        # THE PIN IS READ, NEVER RESTATED. publish-site.yml declares HUGO_VERSION and
        # DOXYGEN_VERSION once each; duplicating either here would fail in the worst direction --
        # bump the workflow and this check starts calling a correct local install wrong. Phase 5
        # exists only to predict the publish, so a check that can disagree with the publish is
        # worse than no check.
        $workflow = Join-Path $repo ".github\workflows\publish-site.yml"

        foreach ($tool in @(@{ n = "hugo"; key = "HUGO_VERSION" }, @{ n = "doxygen"; key = "DOXYGEN_VERSION" })) {
            $want = Select-String -Path $workflow -Pattern "^\s*$($tool.key):\s*([\d.]+)\s*$" |
                    ForEach-Object { $_.Matches.Groups[1].Value } | Select-Object -First 1

            if (-not $want) {
                $findings.Add( "could not read $($tool.key) from $workflow; the version check is unenforced" )
                continue
            }

            $cmd = Get-Command $tool.n -ErrorAction SilentlyContinue

            if (-not $cmd) {
                $findings.Add( "$($tool.n) not installed; phase 5 will be skipped" )
                continue
            }

            $reported = if ($tool.n -eq "hugo") {
                (& hugo version) -replace '^hugo v([\d.]+).*', '$1'
            } else {
                (& doxygen --version) -replace '^([\d.]+).*', '$1'
            }

            Write-Host "  $($tool.n.PadRight(15)) $reported$(if ($reported -ne $want) { "  (publish-site.yml pins $want)" })"

            if ($reported -ne $want) {
                $findings.Add( "$($tool.n) $reported differs from the pinned $want; phase 5 stops predicting the publish" )
            }
        }

        # The staging gate must run the same generator as the deploy, and no workflow can import
        # another's env, so the pairing is only a convention -- checked here because nothing else
        # checks it, and a drift makes web.yml validate a site that is not the one that ships.
        $publishHugo = Select-String -Path $workflow -Pattern "^\s*HUGO_VERSION:\s*([\d.]+)\s*$" |
                       ForEach-Object { $_.Matches.Groups[1].Value } | Select-Object -First 1
        $webHugo = Select-String -Path (Join-Path $repo ".github\workflows\web.yml") -Pattern "^\s*HUGO_VERSION:\s*([\d.]+)\s*$" |
                   ForEach-Object { $_.Matches.Groups[1].Value } | Select-Object -First 1

        if ($publishHugo -and $webHugo -and $publishHugo -ne $webHugo) {
            $findings.Add( "HUGO_VERSION disagrees between workflows: publish-site.yml $publishHugo, web.yml $webHugo" )
        }
    }

    if (Test-PhaseSelected 1) {
        if (-not (Test-Path (Join-Path $verifyBuildDir "CMakeCache.txt"))) {
            $fatal.Add( "$verifyBuildDir is not configured. This script never configures a build directory -- configure it once by hand, then re-run." )
        }
    }

    $drive = Get-PSDrive -Name (Split-Path $repo -Qualifier).TrimEnd(":")
    $freeGb = [math]::Round( $drive.Free / 1GB, 1 )
    Write-Host "  Free on $($drive.Name):       $freeGb GB"

    if ($freeGb -lt $MinimumFreeGb) {
        $fatal.Add( "only $freeGb GB free on $($drive.Name): and $MinimumFreeGb GB is the floor. Refusing to start a multi-hour run that cannot finish." )
    }

    foreach ($f in $findings) { Write-Host "  WARN  $f" -ForegroundColor Yellow }
    foreach ($f in $fatal)    { Write-Host "  STOP  $f" -ForegroundColor Red }

    if ($fatal.Count -gt 0) {
        throw "preflight failed; nothing was built"
    }

    return $version
}

# ---------------------------------------------------------------------------
# Phase runner. Output goes to a per-phase log rather than the console: a full build's
# output is tens of thousands of lines, and the useful thing at the console is a
# result plus, on failure, the tail that explains it.
# ---------------------------------------------------------------------------
$results = [System.Collections.Generic.List[object]]::new()

function Invoke-BuildPhase( [hashtable] $spec, [scriptblock] $body )
{
    Write-Head "Phase $($spec.Id) -- $($spec.Name)"

    $log = Join-Path $LogDir ("phase{0}-{1}.log" -f $spec.Id, ($spec.Name -replace '[^\w]+', '-'))
    Write-Host "  started $(Get-Date -Format 'HH:mm:ss')   log: $log"

    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $ok = $true
    $note = ""

    try {
        & $body 2>&1 | Out-File -FilePath $log -Encoding utf8
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) { $ok = $false; $note = "exit $LASTEXITCODE" }
    }
    catch {
        $ok = $false
        $note = $_.Exception.Message
        $_ | Out-String | Out-File -FilePath $log -Append -Encoding utf8
    }

    $watch.Stop()
    $elapsed = "{0:hh\:mm\:ss}" -f $watch.Elapsed

    $results.Add( [pscustomobject]@{ Phase = $spec.Id; Name = $spec.Name; Ok = $ok; Elapsed = $elapsed; Note = $note; Log = $log } )

    if ($ok) {
        Write-Host "  OK      $elapsed" -ForegroundColor Green
    } else {
        Write-Host "  FAILED  $elapsed  $note" -ForegroundColor Red
        Write-Host "  --- last 40 lines of $log ---" -ForegroundColor Red
        Get-Content $log -Tail 40 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
        throw "phase $($spec.Id) failed; re-run with -Phase $($spec.Id) after fixing, or -Phase $(($spec.Id + 1))+ to continue past it"
    }
}

# ===========================================================================
# Plan / dry run
# ===========================================================================
Write-Head "Mila artifact build"
Write-Host "  This script BUILDS. It publishes nothing -- no PyPI, no Docker Hub, no git, no Pages."
Write-Host ""
Write-Host ("  {0,-3} {1,-38} {2,-8} {3}" -f "#", "Phase", "Where", "Estimate (cold)")

foreach ($p in $phases) {
    $mark = if (Test-PhaseSelected $p.Id) { " " } else { "-" }
    Write-Host ("{0} {1,-3} {2,-38} {3,-8} {4}" -f $mark, $p.Id, $p.Name, $p.Kind, $p.Estimate)
    Write-Host ("      {0}" -f $p.What) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "  Phases run strictly in order. Native and Docker phases are never concurrent:"
Write-Host "  they contend through WSL2, which has cost real time before."

if ($DryRun) {
    Invoke-Preflight | Out-Null
    Write-Host ""
    Write-Host "Dry run: nothing was built, downloaded or published." -ForegroundColor Cyan
    return
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$version = Invoke-Preflight

# CPM re-resolves from GitHub on every --fresh configure, and phase 2 does that once per
# interpreter. A shared source cache turns two downloads per run into zero after the first.
if (-not $env:CPM_SOURCE_CACHE) {
    $env:CPM_SOURCE_CACHE = Join-Path $repo "out\cpm-cache"
    Write-Host "  CPM_SOURCE_CACHE set to $env:CPM_SOURCE_CACHE for this run"
}

$runWatch = [System.Diagnostics.Stopwatch]::StartNew()

# ---------------------------------------------------------------------------
# The native phases need cl.exe. Entered ONCE here rather than per phase: the wheel
# script enters its own shell in its own process scope, and doing it twice in one
# process just prepends PATH again.
# ---------------------------------------------------------------------------
if (@($selected | Where-Object { $_.Kind -eq "native" }).Count -gt 0) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    # Enter-VsDevShell's internals shell out to a BARE `vswhere.exe`, so with the Installer
    # directory off PATH it prints "'vswhere.exe' is not recognized" and carries on -- harmless,
    # because -VsInstallPath below is explicit, but a spurious error line at the top of a build
    # log is how a real one later goes unread. Putting the directory on PATH removes the message
    # rather than hiding it.
    $vswhereDir = Split-Path $vswhere -Parent
    if ((Test-Path $vswhereDir) -and ($env:PATH -notlike "*$vswhereDir*")) {
        $env:PATH = "$vswhereDir;$env:PATH"
    }

    $vsPath = & $vswhere -latest -products * -property installationPath
    Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
    Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments "-arch=x64 -host_arch=x64" -SkipAutomaticLocation | Out-Null
    Set-Location $repo
}

# --- Phase 1 -- library and suite ------------------------------------------
if (Test-PhaseSelected 1) {
    Invoke-BuildPhase ($phases | Where-Object Id -eq 1) {
        # Build only. Configuring would wipe a cache this script does not own.
        cmake --build $verifyBuildDir
        if ($LASTEXITCODE -ne 0) { throw "build failed" }

        ctest --test-dir $verifyBuildDir --output-on-failure
    }
}

# --- Phase 2 -- Windows wheels ---------------------------------------------
if (Test-PhaseSelected 2) {
    Invoke-BuildPhase ($phases | Where-Object Id -eq 2) {
        & (Join-Path $repo "scripts\pypi\build-wheel-windows.ps1")
    }
}

# --- Phase 3 -- Linux wheels -----------------------------------------------
# First Docker phase. Nothing native runs alongside it: the script is serial by design.
if (Test-PhaseSelected 3) {
    Invoke-BuildPhase ($phases | Where-Object Id -eq 3) {
        $compose = Join-Path $repo "Docker\docker-compose.wheel.yml"

        docker compose -f $compose build
        if ($LASTEXITCODE -ne 0) { throw "wheel container build failed" }

        docker compose -f $compose run --rm mila-wheel mila-build-wheel
        if ($LASTEXITCODE -ne 0) { throw "mila-build-wheel failed" }
    }
}

# --- Phase 4 -- container images -------------------------------------------
# build-runtime-image.sh, never publish-image.sh: the latter refuses to run outside a
# pushed tag, and a local tag is what this script is for. Git Bash rewrites the POSIX
# repo path into the Windows form docker needs for the build context, so MSYS_NO_PATHCONV
# is deliberately NOT set here -- see the note in verify-image.sh.
if (Test-PhaseSelected 4) {
    Invoke-BuildPhase ($phases | Where-Object Id -eq 4) {
        $script = "scripts/dockerhub/build-runtime-image.sh"
        $posixRepo = $repo -replace '\\', '/'

        foreach ($target in @("runtime", "devel")) {
            # MILA_CLEAN_BUILD=1 to match what publish-image.sh forces. Validating with a warm
            # cache mount would validate the wrong thing: --no-cache does not clear those mounts,
            # and an image assembled from an earlier tree's objects is the exact failure the
            # publish gates exist to prevent. It costs a full module-graph compile, which both
            # targets share through the builder stage.
            & $gitBash -lc "cd '$posixRepo' && MILA_IMAGE_TARGET=$target MILA_RUNTIME_IMAGE_TAG=mila-llm:local-$target MILA_CLEAN_BUILD=1 $script"
            if ($LASTEXITCODE -ne 0) { throw "$target image build failed" }
        }
    }
}

# --- Phase 5 -- website ----------------------------------------------------
# Mirrors publish-site.yml: Doxygen first (WARN_AS_ERROR is the ratchet that silently
# blocks a publish), then Hugo. Output paths match the workflow so a break here is the
# break CI would have.
if (Test-PhaseSelected 5) {
    $haveSiteTools = (Get-Command hugo -ErrorAction SilentlyContinue) -and (Get-Command doxygen -ErrorAction SilentlyContinue)

    if ($haveSiteTools) {
        Invoke-BuildPhase ($phases | Where-Object Id -eq 5) {
            New-Item -ItemType Directory -Force -Path (Join-Path $repo "build\docs") | Out-Null

            doxygen (Join-Path $repo "Mila\Docs\Doxyfile")
            if ($LASTEXITCODE -ne 0) { throw "Doxygen failed (WARN_AS_ERROR): a doc warning blocks publish-site.yml" }

            hugo --source (Join-Path $repo "Web") --destination (Join-Path $repo "build\site") --minify --gc
            if ($LASTEXITCODE -ne 0) { throw "hugo build failed" }
        }
    } else {
        Write-Host "  Phase 5 skipped: hugo or doxygen not installed" -ForegroundColor Yellow
    }
}

# --- Phase 6 -- inventory --------------------------------------------------
if (Test-PhaseSelected 6) {
    Write-Head "Phase 6 -- Inventory"

    # EVERYTHING HERE IS DATED AGAINST THE RUN START. Reporting what is on disk as though this
    # run produced it is worse than reporting nothing: the first version of this phase announced
    # "Site built at build/site" after a run in which phase 5 never executed, because a directory
    # from an earlier build was still there. An inventory that cannot tell fresh from stale
    # cannot be used to decide a release is ready.
    $wheelDir = Join-Path $repo "out\wheel"
    $wheels = @(Get-ChildItem $wheelDir -Filter "*.whl" -ErrorAction SilentlyContinue)
    $fresh = @($wheels | Where-Object { $_.LastWriteTime -ge $runStart })
    $stale = @($wheels | Where-Object { $_.LastWriteTime -lt $runStart })

    Write-Host "  Wheels in out/wheel: $($wheels.Count) total, $($fresh.Count) from this run"

    foreach ($w in $wheels) {
        $tag = if ($w.LastWriteTime -ge $runStart) { "this run" } else { "PRE-EXISTING $($w.LastWriteTime.ToString('MM-dd HH:mm'))" }
        Write-Host ("    {0,-58} {1,7:N1} MB  {2}" -f $w.Name, ($w.Length / 1MB), $tag)
    }

    # out/wheel is published from ONE glob, so a wheel this run did not build would be uploaded
    # beside the ones it did -- and a PyPI filename cannot be reused or withdrawn.
    if ($stale.Count -gt 0) {
        Write-Host "  WARN  $($stale.Count) wheel(s) predate this run. Publishing globs this directory; remove them before a release build." -ForegroundColor Yellow
    }

    # Four is the published set: two interpreters on two platforms.
    if ((Test-PhaseSelected 2) -and (Test-PhaseSelected 3) -and ($fresh.Count -ne 4)) {
        Write-Host "  WARN  a full wheel run should leave exactly 4 fresh wheels; this left $($fresh.Count)" -ForegroundColor Yellow
    }

    $versions = @($wheels | ForEach-Object { ($_.Name -split '-')[1] } | Sort-Object -Unique)
    if ($versions.Count -gt 1) {
        Write-Host "  WARN  wheels carry more than one version: $($versions -join ', ')" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "  Local images:"
    $images = docker images --filter "reference=mila-llm:local-*" --format "{{.Repository}}:{{.Tag}}|{{.Size}}|{{.CreatedAt}}" 2>$null

    if (-not $images) { Write-Host "    (none)" }

    foreach ($line in $images) {
        $parts = $line -split '\|'
        $created = [datetime]::Parse( ($parts[2] -replace ' [A-Z]{3,4}$', '') )
        $tag = if ($created -ge $runStart) { "this run" } else { "PRE-EXISTING $($created.ToString('MM-dd HH:mm'))" }
        Write-Host ("    {0,-34} {1,10}  {2}" -f $parts[0], $parts[1], $tag)
    }

    $siteIndex = Join-Path $repo "build\site\index.html"

    if (Test-Path $siteIndex) {
        $built = (Get-Item $siteIndex).LastWriteTime
        $tag = if ($built -ge $runStart) { "built by this run" } else { "PRE-EXISTING from $($built.ToString('MM-dd HH:mm'))" }
        Write-Host ""
        Write-Host "  Site at build/site: $tag"
    }
}

# ===========================================================================
# Summary. The elapsed column is the point: it replaces this script's own cold
# estimates with what the phases actually cost on this machine.
# ===========================================================================
$runWatch.Stop()

Write-Head "Summary"
$results | Format-Table -AutoSize @{ n = "#"; e = { $_.Phase } }, Name, @{ n = "Result"; e = { if ($_.Ok) { "ok" } else { "FAILED" } } }, Elapsed

Write-Host ("  Total {0:hh\:mm\:ss}   version {1}" -f $runWatch.Elapsed, $version)
Write-Host "  Logs: $LogDir"
Write-Host ""
Write-Host "  Nothing was published. To release, follow RELEASING.md -- these artifacts are not it." -ForegroundColor Cyan
