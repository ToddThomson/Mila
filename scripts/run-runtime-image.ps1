# Run the locally built Mila image the way a user of the published one would.
#
# The store lives in a named docker volume rather than the repo, because that is what a
# user without a checkout has -- and it is what keeps a ~6 GB model install from being
# discarded by --rm. Chat opens on an empty store: install with `/install <name>`.
#
#   .\scripts\run-runtime-image.ps1                  # Chat (default)
#   .\scripts\run-runtime-image.ps1 serve            # MIS, port published to the host
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args_ = @()
)

$tag = if ($env:MILA_RUNTIME_IMAGE_TAG) { $env:MILA_RUNTIME_IMAGE_TAG } else { "mila-llm:local" }
$publish = if ($Args_.Count -gt 0 -and $Args_[0] -eq "serve") { @("--publish", "6452:6452") } else { @() }

docker run --rm -it --gpus all `
    -v mila-store:/models `
    @publish `
    $tag @Args_
