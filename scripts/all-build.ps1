# Configure + build the full Mila product set (library, samples, Chat, Python binding)
# in the Mila container (writes to the mila-build volume).
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
cd "$scriptDir\..\Docker"
docker compose run --rm mila-dev mila-build-all
