# Mila Network Viewer

A Windows 11 application for visualizing and analyzing Mila neural network models.

## Features

- **3-Pane Layout**
  - Left: Hierarchical tree view of network modules
  - Right-Top: Document view showing module details and architecture
  - Right-Bottom: Tabbed output window for logs and validation results

- **Model Visualization**
  - Browse network structure and module hierarchy
  - Inspect module configurations and parameter counts
  - View layer connections and data flow

- **Tools**
  - Model validation and integrity checking
  - Architecture export to documentation
  - Recent files support

## Building

cd Tools/MilaViewer cmake -B build -G Ninja cmake --build build --config Release

## Requirements

- Windows 11 (or Windows 10 version 2004+)
- Visual Studio 2022 or later
- CMake 3.28+
- Mila framework

## Usage

1. Launch MilaViewer.exe
2. Open a Mila model file (.mila) via File → Open
3. Explore the network structure in the tree view
4. Select modules to view their details

## Architecture

The application uses C++23 modules for clean separation:

- `MilaViewer.Application` - Main application lifecycle
- `MilaViewer.MainWindow` - Main window and layout management  
- `MilaViewer.ModelTreeView` - Network hierarchy tree
- `MilaViewer.DocumentView` - Module detail display
- `MilaViewer.OutputView` - Tabbed output logs

All UI is native Win32 for maximum performance and Windows 11 integration.
