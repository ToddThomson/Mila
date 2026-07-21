# Build the Mila Inference Server (mila binding + Python server venv) in the container.
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
cd "$scriptDir\..\Docker"
docker compose run --rm mila-dev mila-build-mis
