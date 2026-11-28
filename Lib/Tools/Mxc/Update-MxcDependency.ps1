# Updates Lib/Deps/mxc to one MXC release, run once per Mila release (ContainedPrograms.md 4.10, decision 13).
#
# Windows: mxc_ffi.dll built here from MXC's source at the release tag, unrouted -- Microsoft's signed telemetry
# component is removed from the checkout before the build, and the build is refused unless MXC's generated provider
# says IS_UTC_ROUTED is false. macOS and Linux: the libraries from Microsoft's release archive, which carry no
# telemetry. Each file's SHA-256 is written to Lib/Deps/mxc/SOURCE.md.
#
# Needs git, the Visual Studio developer environment (MSVC's linker) and rustup's cargo on PATH; MXC pins its own
# Rust toolchain (src/rust-toolchain.toml), which rustup installs on first use.
#
#   Lib/Tools/Mxc/Update-MxcDependency.ps1 -Version 1.1.0 -WorkFolder D:\Work\mxc

param(
    [Parameter( Mandatory )] [string] $Version,
    [Parameter( Mandatory )] [string] $WorkFolder,
    [string] $ReleaseArchive
)

$ErrorActionPreference = 'Stop'

$repository = Resolve-Path ( Join-Path $PSScriptRoot '..\..\..' )
$deps = Join-Path $repository 'Lib\Deps\mxc'
$tag = "v$Version"
$source = Join-Path $WorkFolder 'source'

foreach ( $tool in 'git', 'cargo', 'link' ) {
    if ( -not ( Get-Command $tool -ErrorAction SilentlyContinue ) ) {
        throw "$tool is not on PATH: run from a Visual Studio developer shell with rustup's cargo on PATH."
    }
}

New-Item -ItemType Directory -Force $WorkFolder | Out-Null

if ( Test-Path $source ) {
    Remove-Item -LiteralPath $source -Recurse -Force
}

git -c advice.detachedHead=false clone --quiet --depth 1 --branch $tag https://github.com/microsoft/mxc.git $source
if ( $LASTEXITCODE -ne 0 ) { throw "Cannot clone MXC at $tag." }

$commit = ( git -C $source rev-parse HEAD ).Trim()

# Without the signed component, a release build has no routing group. MXC_TELEMETRY_COMPONENT_PATH cannot carry an
# empty value from PowerShell, and an unset one selects this bundled component, so it is removed instead.
$component = Join-Path $source 'src\mxc-sdk\resources\telemetry-component'

if ( Test-Path $component ) {
    Remove-Item -LiteralPath $component -Recurse -Force
}

Remove-Item Env:\MXC_TELEMETRY_COMPONENT_PATH -ErrorAction SilentlyContinue

$target = 'x86_64-pc-windows-msvc'
Push-Location ( Join-Path $source 'src' )

try {
    cargo build --release --target $target -p mxc_ffi
    if ( $LASTEXITCODE -ne 0 ) { throw 'cargo build failed.' }
} finally {
    Pop-Location
}

$release = Join-Path $source "src\target\$target\release"
$providers = @( Get-ChildItem ( Join-Path $release 'build' ) -Recurse -Filter provider_def.rs )

if ( $providers.Count -ne 1 ) {
    throw "Expected one generated provider_def.rs, found $( $providers.Count )."
}

if ( ( Get-Content $providers[ 0 ].FullName -Raw ) -notmatch 'IS_UTC_ROUTED:\s*bool\s*=\s*false' ) {
    throw 'MXC generated a routed provider: the library would send telemetry, so it is not copied.'
}

if ( -not $ReleaseArchive ) {
    $ReleaseArchive = Join-Path $WorkFolder 'mxc-release-binaries.zip'

    if ( -not ( Test-Path $ReleaseArchive ) ) {
        gh release download $tag --repo microsoft/mxc --pattern 'mxc-release-binaries.zip' --dir $WorkFolder
        if ( $LASTEXITCODE -ne 0 ) { throw "Cannot download MXC's $tag release archive." }
    }
}

$extracted = Join-Path $WorkFolder 'release'

if ( Test-Path $extracted ) {
    Remove-Item -LiteralPath $extracted -Recurse -Force
}

Expand-Archive -Path $ReleaseArchive -DestinationPath $extracted

$files = [ordered]@{
    'windows-x64\mxc_ffi.dll' = @{ From = Join-Path $release 'mxc_ffi.dll'; Origin = "built from source at $commit, unrouted" }
    'linux-x64\libmxc_ffi.so' = @{ From = Join-Path $extracted 'x64\libmxc_ffi.so'; Origin = "Microsoft's $tag release" }
    'linux-arm64\libmxc_ffi.so' = @{ From = Join-Path $extracted 'arm64\libmxc_ffi.so'; Origin = "Microsoft's $tag release" }
    'macos-arm64\libmxc_ffi.dylib' = @{ From = Join-Path $extracted 'arm64\libmxc_ffi.dylib'; Origin = "Microsoft's $tag release" }
}

if ( Test-Path $deps ) {
    Remove-Item -LiteralPath $deps -Recurse -Force
}

New-Item -ItemType Directory -Force $deps | Out-Null
Copy-Item ( Join-Path $source 'LICENSE.md' ) ( Join-Path $deps 'LICENSE.md' )

$rows = foreach ( $name in $files.Keys ) {
    $file = $files[ $name ]

    if ( -not ( Test-Path $file.From ) ) {
        throw "Missing $( $file.From )."
    }

    $destination = Join-Path $deps $name
    New-Item -ItemType Directory -Force ( Split-Path $destination ) | Out-Null
    Copy-Item $file.From $destination
    $hash = ( Get-FileHash $destination -Algorithm SHA256 ).Hash.ToLower()

    "| ``$( $name -replace '\\', '/' )`` | $( $file.Origin ) | ``$hash`` |"
}

$sourceText = @(
    "# MXC $Version"
    ''
    "Written by ``Lib/Tools/Mxc/Update-MxcDependency.ps1``. MXC is Microsoft's, MIT-licensed (``LICENSE.md``):"
    "https://github.com/microsoft/mxc, tag ``$tag``, commit ``$commit``. Loaded at run time, never linked"
    '(`Specifications/ContainedPrograms.md` 4.10).'
    ''
    '| File | Origin | SHA-256 |'
    '|---|---|---|'
    $rows
)

Set-Content -Path ( Join-Path $deps 'SOURCE.md' ) -Value $sourceText -Encoding utf8NoBOM
Get-Content ( Join-Path $deps 'SOURCE.md' )
