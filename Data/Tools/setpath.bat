@echo off
REM setpath.bat - Add Mila tokenize tools to PATH for current session

setlocal enabledelayedexpansion

set "TOOLS_DIR=%~dp0"
set "DEBUG_PATH=%TOOLS_DIR%Debug"
set "RELEASE_PATH=%TOOLS_DIR%Release"

REM Check and add Debug path
if exist "%DEBUG_PATH%" (
    echo %PATH% | find /i "%DEBUG_PATH%" >nul
    if errorlevel 1 (
        set "PATH=%DEBUG_PATH%;%PATH%"
        echo Added to PATH: %DEBUG_PATH%
    ) else (
        echo Already in PATH: %DEBUG_PATH%
    )
) else (
    echo Debug directory not found: %DEBUG_PATH%
)

REM Check and add Release path
if exist "%RELEASE_PATH%" (
    echo %PATH% | find /i "%RELEASE_PATH%" >nul
    if errorlevel 1 (
        set "PATH=%RELEASE_PATH%;%PATH%"
        echo Added to PATH: %RELEASE_PATH%
    ) else (
        echo Already in PATH: %RELEASE_PATH%
    )
) else (
    echo Release directory not found: %RELEASE_PATH%
)

echo.
echo Tokenize tools are now available. Try: tokenize --help
echo Note: This PATH change is only for the current command prompt session.
echo.

endlocal & set "PATH=%PATH%"