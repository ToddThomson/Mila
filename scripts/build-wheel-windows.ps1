# Build the mila-llm Windows wheel for PyPI. Writes to out/wheel.
#
# The counterpart to Docker/build-wheel.sh, and deliberately the same shape: configure
# and build come from a CMake preset (x64-wheel), and this script only does the part
# CMake does not -- package. There is no repair step; unlike auditwheel on Linux, nothing
# needs vendoring here because the CUDA DLLs arrive from the nvidia-* wheels and
# mila/__init__.py registers their directories before the extension loads.
#
# Run it from anywhere; it locates the repository from its own path.
$ErrorActionPreference = "Stop"

$repo = Split-Path -Path (Split-Path -Path $MyInvocation.MyCommand.Path -Parent) -Parent
$build = Join-Path $repo "out\build\x64-wheel"
$stage = Join-Path $repo "out\wheel-stage"
$out = Join-Path $repo "out\wheel"

# One wheel per interpreter: pybind11 cannot use Py_LIMITED_API, so there is no abi3
# build covering a range. These must match requires-python in pyproject.toml -- metadata
# promising an interpreter with no wheel behind it turns a clear "requires a different
# Python" into "no matching distribution".
$interpreters = @("3.12", "3.13")

$pythons = foreach ($version in $interpreters) {
    $found = (& py "-$version" -c "import sys; print(sys.executable)" 2>$null)

    if (-not $found) {
        throw "Python $version not found. Install it, or drop it from `$interpreters and from requires-python."
    }

    [pscustomobject]@{ Version = $version; Path = $found }
}

# MSVC is not on PATH in a plain shell, and the C++23 module units need cl.exe. Enter the
# VS developer environment in-process so cmake inherits it.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -property installationPath

Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments "-arch=x64 -host_arch=x64" -SkipAutomaticLocation | Out-Null

Set-Location $repo

# Clear THIS platform's wheels ONCE, before the loop, and never the whole directory: the
# Linux wheel lands here too and both are published from one glob. A previous run's wheel
# carries a different version, and leaving one behind is how a stale build gets uploaded
# alongside the intended one -- which cannot be undone on PyPI. Inside the loop this would
# delete the wheel the previous interpreter just built.
New-Item -ItemType Directory -Force -Path $out | Out-Null
Get-ChildItem $out -Filter "mila_llm-*win_amd64.whl" -ErrorAction SilentlyContinue | Remove-Item -Force

foreach ($interpreter in $pythons) {
    $python = $interpreter.Path
    Write-Host "`n=== Python $($interpreter.Version) -- $python ===" -ForegroundColor Cyan

    # PYBIND11_FINDPYTHON=NEW and Python_EXECUTABLE are what actually select the
    # interpreter. pybind11 v3 resolves through FindPython, so Python3_EXECUTABLE alone
    # steers only the rest of the project: a tree configured with Python3_EXECUTABLE=3.12
    # was measured building _mila.cp313. That mismatch is silent and ships -- the wheel
    # tag comes from the packaging venv below, so the archive would be tagged for one
    # interpreter and contain an extension for another, and `import mila` then fails on a
    # machine that has only the tagged one. Python3_EXECUTABLE stays for the rest of the
    # build; all three are passed so nothing resolves independently.
    # --fresh because FindPython caches the include and library paths it resolved, and
    # those do not move when the executable does: reconfiguring a tree built for one
    # interpreter against another fails with "Could NOT find Python (missing:
    # Development.Module)" while reporting the NEW version as found. One build tree, wiped
    # per interpreter, rather than a tree each -- a wheel build is a release-time operation,
    # so predictability is worth more than incrementality here.
    cmake --preset x64-wheel --fresh `
        "-DPYBIND11_FINDPYTHON=NEW" `
        "-DPython_EXECUTABLE=$python" `
        "-DPython3_EXECUTABLE=$python"
    if ($LASTEXITCODE -ne 0) { throw "configure failed for Python $($interpreter.Version)" }

    cmake --build $build
    if ($LASTEXITCODE -ne 0) { throw "build failed for Python $($interpreter.Version)" }

    # Package from a COPY, never from the package tree in place. MilaPy's POST_BUILD
    # stages into Mila/Bindings/Package/src/mila, which accumulates: a Linux build leaves
    # its _mila*.so and every interpreter leaves its own .pyd. package-data globs both, so
    # packaging in place would put every extension ever built into every wheel. Copying
    # also means this never deletes anything from the working tree.
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    Copy-Item -Recurse (Join-Path $repo "Mila\Bindings\Package") $stage

    # Take the extension from THIS build rather than trusting what POST_BUILD left behind:
    # strip every staged extension, then copy in the one just produced. Deterministic
    # regardless of what earlier runs deposited in the working tree.
    Get-ChildItem (Join-Path $stage "src\mila") -Filter "_mila*" | Remove-Item -Force

    # Named by the interpreter's OWN suffix, never "the .pyd in the build directory":
    # --fresh clears the CMake cache but not build outputs, so a previous interpreter's
    # extension is still sitting there. Matching the tag exactly is what stops a stale
    # artifact being packaged as this one.
    $suffix = (& $python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
    $built = Join-Path $build "Mila\Bindings\_mila$suffix"

    if (-not (Test-Path $built)) {
        throw "expected $built from the Python $($interpreter.Version) build, and it is not there"
    }

    Copy-Item $built (Join-Path $stage "src\mila")

    # Packaging tools in their own environment, so this never installs into the user's
    # interpreter -- and one venv PER interpreter, because pip wheel takes the wheel's
    # cp tag from the interpreter running it. setuptools is named explicitly: since 3.12
    # ensurepip no longer seeds it, and pyproject.toml declares it as the build backend.
    $venv = Join-Path $repo "out\wheel-venv-$($interpreter.Version)"

    if (-not (Test-Path $venv)) {
        & $python -m venv $venv
    }

    $venvPython = Join-Path $venv "Scripts\python.exe"
    & $venvPython -m pip install --quiet --upgrade pip "setuptools>=77" wheel

    & $venvPython -m pip wheel --no-deps --no-build-isolation -w $out $stage
    if ($LASTEXITCODE -ne 0) { throw "pip wheel failed for Python $($interpreter.Version)" }

    Remove-Item -Recurse -Force $stage
}

Write-Host "`nWheels written to $out :"
Get-ChildItem $out -Filter "*.whl" | ForEach-Object { "  {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB) }
