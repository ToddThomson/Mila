# setpath.ps1 - Add Mila tokenize tools to PATH for current session

$ToolsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DebugPath = Join-Path $ToolsDir "Debug"
$ReleasePath = Join-Path $ToolsDir "Release"

# Check if directories exist
$pathsToAdd = @()
if (Test-Path $DebugPath) {
    $pathsToAdd += $DebugPath
}
if (Test-Path $ReleasePath) {
    $pathsToAdd += $ReleasePath
}

if ($pathsToAdd.Count -eq 0) {
    Write-Host "No tool directories found. Build the project first." -ForegroundColor Yellow
    exit 1
}

# Add to current session PATH
foreach ($path in $pathsToAdd) {
    if ($env:Path -notlike "*$path*") {
        $env:Path = "$path;$env:Path"
        Write-Host "Added to PATH: $path" -ForegroundColor Green
    } else {
        Write-Host "Already in PATH: $path" -ForegroundColor Cyan
    }
}

Write-Host "`nTokenize tools are now available. Try: tokenize --help" -ForegroundColor Green
Write-Host "Note: This PATH change is only for the current PowerShell session." -ForegroundColor Yellow