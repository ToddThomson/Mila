# Run the built Chat app in the Mila container (needs GPU + weights under Data/Models).
# Any arguments are forwarded to ChatApp (e.g. --config Data/other.json, --help).
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
cd "$scriptDir\..\Docker"
docker compose run --rm mila-dev mila-chat @args
