# Chat Desktop

Chat's window: a page (`Ui/`) over Chat's session, hosted by an Electron window (`Window/`).
The design of record is `Mila/Specifications/ChatDesktop.md`. Until the session process exists
(phase 3), the window runs the mock session (`Ui/mock/`), which plays scripted scenarios.

## The page in a browser

`Ui/` is plain HTML, CSS and JavaScript modules with no build step. With no window around it, the page
runs over the in-page mock. Modules do not load from `file://`, so serve the folder:

```
cd Mila/Apps/Chat/Desktop/Ui
python -m http.server 8765 --bind 127.0.0.1
```

and open `http://127.0.0.1:8765/?scenario=1`. The scenario menu at the foot of the sidebar switches
scenarios and plays one; `&autoplay=1` plays it on load.

## The window

Node 22 or later. Electron and electron-builder are pinned in `Window/package.json`, with the lock file.

```
cd Mila/Apps/Chat/Desktop/Window
npm ci
npm start -- --scenario=6
```

`--scenario=<n>` picks the mock's scenario. `MILA_CHAT_SESSION=<path>` starts that program as the session
instead of the mock. The window serves the page at `mila://chat/`, starts the session as a child process
speaking JSON lines on stdio, and relays messages between them; a link the page asks to open goes to the
system browser.

The mock session also runs on its own, for any client: `node Ui/mock/stdio.js --scenario 1`.

## Tests

```
cd Mila/Apps/Chat/Desktop/Ui
node --test

cd ../Window
node --test
```

The first covers the page's state, its markdown renderer and the mock's scenarios; the second drives the
mock session over stdio.

## Builds

| | Command, in `Window/` | Output in `Window/dist/` |
|---|---|---|
| Windows installer, on Windows | `npm ci` then `npm run build:windows` | `Mila-Chat-Setup-<version>.exe` and `win-unpacked/` |
| Windows installer, on Linux | the same, with 64- and 32-bit Wine installed (`wine64`, `wine32:i386`) | the same |
| Linux folder | `npm run build:linux` | `linux-unpacked/` |

Builds are unsigned. The icon is the Mila mark, in `Window/icons/` (electron-builder's build resources),
rendered from the website's `mila-mark.svg`. A packaged build has no developer tools and no menu bar on
Windows and Linux.

Electron downloads its binary the first time `npm start` runs, and electron-builder downloads NSIS on the
first build. Both cache per user; `ELECTRON_CACHE` and `ELECTRON_BUILDER_CACHE` move them.

## CI

`.github/workflows/desktop.yml` runs on any change under this folder: both test suites on Linux, then the
Windows installer built from the lock file, kept as a workflow upload for 14 days. It is not a release;
until the session process exists the installer runs the mock.
