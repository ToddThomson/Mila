# Chat Desktop

Chat's window: a page (`Ui/`) over Chat's session, hosted by an Electron window (`Window/`, phase 1b).
The design of record is `Mila/Specifications/ChatDesktop.md`.

## The page against the mock

`Ui/` is plain HTML, CSS and JavaScript modules with no build step. With no window around it, the page
runs over the in-page mock session (`Ui/mock/`). Modules do not load from `file://`, so serve the folder:

```
cd Mila/Apps/Chat/Desktop/Ui
python -m http.server 8765 --bind 127.0.0.1
```

and open `http://127.0.0.1:8765/?scenario=1`. The scenario menu at the foot of the sidebar switches
scenarios and plays one; `&autoplay=1` plays it on load.

## Tests

```
cd Mila/Apps/Chat/Desktop/Ui
node --test
```

Node 22 or later; nothing to install.
