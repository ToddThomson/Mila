# Run the Mila Inference Server in the container, publishing its port to the host so a
# harness on the host (or WSL) can drive it. Override port/protocol via env, e.g.:
#   $env:MILA_PROTOCOL="anthropic"; scripts\mis-run.ps1
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
cd "$scriptDir\..\Docker"
$port  = if ($env:MILA_PORT)     { $env:MILA_PORT }     else { "6452" }
$proto = if ($env:MILA_PROTOCOL) { $env:MILA_PROTOCOL } else { "openai" }
docker compose run --rm `
    --publish "${port}:${port}" `
    -e MILA_PORT="$port" `
    -e MILA_PROTOCOL="$proto" `
    mila-dev mila-mis
