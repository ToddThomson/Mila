# Build the mila-llm Linux wheel for PyPI. Writes to out/wheel.
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
cd "$scriptDir\..\Docker"
docker compose -f docker-compose.wheel.yml build
docker compose -f docker-compose.wheel.yml run --rm mila-wheel mila-build-wheel
