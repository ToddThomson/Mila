# Mila Data Tools

This directory contains build tools for processing Mila project assets located in the `/Data` directory.

## Available Tools

- **tokenize.exe** - Processes and tokenizes Mila data assets

## Setup

Before using the tools, you need to add them to your PATH for the current terminal session.

### PowerShell

```powershell
# From the project root
.\Data\tools\setpath.ps1

# Or from this directory
.\setpath.ps1
```

### Command Prompt

```cmd
REM From the project root
Data\tools\setpath.bat

REM Or from this directory
setpath.bat
```

> **Note:** PATH changes are temporary and only affect the current terminal session. You'll need to re-run the setup script each time you open a new terminal.

## Using the Tools

After running the setup script, the tools are available from any directory:

### Tokenize

```powershell
# Basic usage
tokenize <input-file>

# Process multiple files
tokenize *.txt

# From the Data directory
cd Data
tokenize assets\myfile.dat
```

## Build Configurations

Both Debug and Release configurations are available:

- **Debug**: `Data/tools/Debug/tokenize.exe` - Built with debug symbols for development
- **Release**: `Data/tools/Release/tokenize.exe` - Optimized build for production use

When both are in PATH, Release takes precedence. To explicitly use Debug:

```powershell
# PowerShell
& "$PSScriptRoot\Debug\tokenize.exe" myfile.txt

# Command Prompt
Debug\tokenize.exe myfile.txt
```

## Building the Tools

The tools are built automatically when you build the Mila project in Visual Studio or via CMake:

```powershell
# CMake build
cmake --build build --config Release
cmake --build build --config Debug
```

After building, the executables will be in their respective configuration directories under `Data/tools/`.

## Troubleshooting

**"tokenize is not recognized as a command"**
- Make sure you've run the appropriate setup script (`setpath.ps1` or `setpath.bat`)
- Verify the tool has been built (check that `Debug/tokenize.exe` or `Release/tokenize.exe` exists)

**"No tool directories found"**
- Build the project first in Visual Studio or using CMake
- The tools are generated during the build process

**PATH changes don't persist**
- This is expected behavior. The scripts only modify PATH for the current session
- Re-run the setup script in each new terminal window
- For permanent PATH changes, manually add the tool directories to your system PATH via Windows Settings