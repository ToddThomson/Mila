# Mila Visualization (WIP)

This directory contains the visualization subsystem for inspecting model activations, attention maps and module internals.

Status

- Work in progress (WIP): API and renderers are experimental. The API is only for internal development use at this time.
- Current pieces: core context and adapters (`VisualizerContext`), basic renderers and visualizer modules under `Rendering/` and `Modules/`.

Usage
- The visualization API is intended to expose `Mila::Dnn::ITensor` views (see `Core/VisualizerContext.ixx`) so renderers can consume model snapshots.
- A lightweight demo app lives in `Samples/Viz` and demonstrates Direct2D-based rendering; use it as a reference harness.