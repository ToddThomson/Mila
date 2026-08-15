# Build the PUBLISHED Mila image (Docker/Dockerfile.runtime), tagged for local testing.
# The build context is the repo root -- .dockerignore keeps Data/Models, out/ and .git out.
# Override the GPU compatibility list with -Arch, e.g. -Arch "89" for a fast single-arch
# test build. Publishing to a registry is a separate step; this only produces mila-llm:local.
# -Clean discards the cached build tree first. A publish build must use it: `--no-cache`
# invalidates layers but leaves BuildKit cache mounts intact, so without it an image can be
# assembled from objects left by a different source tree.
# -Target selects which published image to build: `runtime` (Chat + the inference server,
# no compiler) or `devel` (that build plus the toolchain, source and build tree). They come
# from one Dockerfile and share the compile.
param(
    [string]$Tag = "mila-llm:local",
    [string]$Arch = "89;90;120",
    [ValidateSet("runtime", "devel")]
    [string]$Target = "runtime",
    [switch]$Clean
)

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$repoRoot = (Resolve-Path "$scriptDir\..").Path

docker build `
    -f "$repoRoot\Docker\Dockerfile.runtime" `
    --target $Target `
    --build-arg MILA_IMAGE_CUDA_ARCHITECTURES="$Arch" `
    --build-arg MILA_CLEAN_BUILD=$(if ($Clean) { "1" } else { "0" }) `
    -t $Tag `
    $repoRoot
