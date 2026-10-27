# Mila Desktop

Mila, the desktop app: a page (`Ui/`) over Chat's session, hosted by an Electron window (`Window/`),
which starts the session process (`Session/`). The design of record is `Mila/Specifications/MilaApp.md`.
For development the window can run the mock session (`Ui/mock/`) instead, which plays scripted scenarios.

## Building the app

Build the `mila-app` target, in Visual Studio from the CMake targets view like any other. It builds the session,
stages it into `Window/` and packages the window, giving `Window/dist/Mila-Setup-<version>.exe`, the installer a user
runs, and `Window/dist/win-unpacked/Mila.exe`, the same app without installing. It is not part of a full build: it
takes about a minute and the installer is about 525 MB. It needs Node 22 or later with `npm` on the PATH when the
build directory is configured, and installs the window's packages the first time. The sections below are its steps,
for running any one of them alone.

## The session process

`mila-session` is built with the C++ targets, beside `mila-chat` in the build directory. It reads
requests as JSON lines on standard input and writes events as JSON lines on standard output (section 4 of the
spec); log records go to standard error, and end of input ends it. Settings resolve as `mila-chat`'s do, from
the same files; `--help` lists its flags.

```
mila-session --model gemma-4-12b-it-fp4 --system-prompt tools-weather
{"v": 1, "type": "ready"}
```

The window starts it with no arguments, so it opens on the model last chosen and takes its settings from the
user's `chat.json`.

### Staging it for the window

The window starts the copy in `Window/session/`, and the package carries that folder. Install the session's
component from a Release build directory into it -- the program, its prompts, the CUDA libraries it loads,
copied from the build's toolkit, and the Visual C++ runtime. The build directory must be configured with
Visual Studio's CMake, or another that knows the VS 2026 toolset; an older one warns and leaves the runtime out.

```
cmake --install out/build/x64-release --component mila-session --prefix Mila/Apps/Desktop/Window/session
```

## The page in a browser

`Ui/` is plain HTML, CSS and JavaScript modules with no build step. With no window around it, the page
runs over the in-page mock. Modules do not load from `file://`, so serve the folder:

```
cd Mila/Apps/Desktop/Ui
python -m http.server 8765 --bind 127.0.0.1
```

and open `http://127.0.0.1:8765/?scenario=1`. The scenario menu at the foot of the sidebar switches
scenarios and plays one; `&autoplay=1` plays it on load.

## The window

Node 22 or later. Electron and electron-builder are pinned in `Window/package.json`, with the lock file.

```
cd Mila/Apps/Desktop/Window
npm ci
npm start
```

`npm start` runs the staged session. `npm start -- --scenario=6` runs the mock's scenario 6 instead, and
`MILA_SESSION=<path>` starts that program as the session. The window serves the page at `mila://chat/`, starts
the session as a child process speaking JSON lines on stdio, and relays messages between them; a link the page
asks to open goes to the system browser.

The mock session also runs on its own, for any client: `node Ui/mock/stdio.js --scenario 1`.

## Tests

```
cd Mila/Apps/Desktop/Ui
node --test

cd ../Window
node --test
```

The first covers the page's state, its markdown renderer and the mock's scenarios; the second drives the
mock session over stdio.

## Builds

| | Command, in `Window/` | Output in `Window/dist/` |
|---|---|---|
| Windows installer, on Windows | the `mila-app` target, or `npm ci` then `npm run build:windows` | `Mila-Setup-<version>.exe` and `win-unpacked/` |
| Windows installer, on Linux | the same, with 64- and 32-bit Wine installed (`wine64`, `wine32:i386`) | the same |
| Linux folder | the `mila-app` target, or `npm run build:linux` | `linux-unpacked/` |

Each npm build needs the session staged first, built for the platform it packages, and refuses to run without it;
`mila-app` stages it afresh each time. The package
carries it as `resources/Session/` and starts it; the mock is not packaged.

Builds are unsigned. The icon is the Mila mark, in `Window/icons/` (electron-builder's build resources),
rendered from the website's `mila-mark.svg`. A packaged build has no developer tools and no menu bar on
Windows and Linux.

Electron downloads its binary the first time `npm start` runs, and electron-builder downloads NSIS on the
first build. Both cache per user; `ELECTRON_CACHE` and `ELECTRON_BUILDER_CACHE` move them.

## CI

`.github/workflows/desktop.yml` runs both test suites on Linux on any change under this folder. It builds no
installer: an installer carries the session, which needs the C++ build.
