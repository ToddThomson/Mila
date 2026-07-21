# Configure + build the Chat app in the Mila container (writes to the mila-build volume).
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
cd "$scriptDir\..\Docker"
docker compose run --rm mila-dev mila-build-chat
