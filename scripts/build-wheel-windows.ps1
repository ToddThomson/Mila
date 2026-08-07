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

# 3.13 explicitly, never "whatever python is on PATH". requires-python is >=3.13,<3.14,
# so a build against 3.12 produces a cp312 wheel that pip will refuse to install against
# its own metadata. Passed to CMake rather than pinned in the preset, because the path is
# per-machine and the preset is committed.
$python = (& py -3.13 -c "import sys; print(sys.executable)")

if (-not $python) {
    throw "Python 3.13 not found. The wheel is cp313; install it or adjust this script."
}

# MSVC is not on PATH in a plain shell, and the C++23 module units need cl.exe. Enter the
# VS developer environment in-process so cmake inherits it.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -property installationPath

Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments "-arch=x64 -host_arch=x64" -SkipAutomaticLocation | Out-Null

Set-Location $repo

cmake --preset x64-wheel "-DPython3_EXECUTABLE=$python"
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

cmake --build $build
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# Package from a COPY, never from the package tree in place. MilaPy's POST_BUILD stages
# the extension into Mila/Bindings/Package/src/mila, where a Linux wheel build has
# already left its _mila*.so -- and package-data globs both *.pyd and *.so, so packaging
# in place would put the Linux extension inside the Windows wheel. Copying also means
# this never deletes anything from the working tree.
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
Copy-Item -Recurse (Join-Path $repo "Mila\Bindings\Package") $stage
Get-ChildItem (Join-Path $stage "src\mila") -Filter "_mila*.so" | Remove-Item -Force

# Packaging tools in their own environment, so this never installs into the user's
# interpreter. setuptools is named explicitly: since 3.12 ensurepip no longer seeds it,
# and pyproject.toml declares it as the build backend.
$venv = Join-Path $repo "out\wheel-venv"

if (-not (Test-Path $venv)) {
    & $python -m venv $venv
}

$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --quiet --upgrade pip "setuptools>=77" wheel

# Clear THIS platform's wheels only, never the whole directory: the Linux wheel lands
# here too and both are published from one glob. A previous run's wheel carries a
# different version, and leaving one behind is how a stale build gets uploaded alongside
# the intended one -- which cannot be undone on PyPI.
New-Item -ItemType Directory -Force -Path $out | Out-Null
Get-ChildItem $out -Filter "mila_llm-*win_amd64.whl" -ErrorAction SilentlyContinue | Remove-Item -Force

& $venvPython -m pip wheel --no-deps --no-build-isolation -w $out $stage
if ($LASTEXITCODE -ne 0) { throw "pip wheel failed" }

Remove-Item -Recurse -Force $stage

Write-Host "`nWheel written to $out :"
Get-ChildItem $out -Filter "*.whl" | ForEach-Object { "  {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB) }
