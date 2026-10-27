# Apps

What Mila ships to the people who run it, built on the library. Each app is a maintained
product, not a demo (teaching code lives under `Mila/Samples`), and each adds no capability of
its own: a gap between what the library can do and what an app reaches is a defect in the app.

- **[Chat](Chat/)** (`mila-chat`) closes the generation loop in-process with a **human** in the
  gate. An instruct chat console with channel-aware streaming, tool calling and model
  hot-switching over any installed instruct model. CUDA-only today.
- **[Assistant](Assistant/)** is Mila, the assistant app: conversations, models and tool calls in a
  window, over a `Mila::AI` agent its session process runs on real models.
- **[Server](Server/)** (`mila-server`) is the Mila Inference Server: it serves the loop over
  OpenAI- and Anthropic-compatible HTTP, so a foreign harness (Codex, Claude Code) drives Mila
  from another process. It is Python, installs as `mila-llm-server`, and imports the `mila`
  binding built from [Mila/Bindings](../Bindings/). How to point the harnesses at it is in
  [Server/docs/harnesses.md](Server/docs/harnesses.md).
- **[Tools/Store](Tools/Store/)** (`mila-store`) installs, lists and removes the models the apps
  run, from the shell.

Every command a user types is lowercase with hyphens and the `mila-` prefix, the same on every
platform. Developer tooling that never ships lives in `Mila/Tools`, not here. The positioning is
in [MilaProductFamily.md](../Specifications/MilaProductFamily.md); the layout and names were
decided in [MilaCommandLineReview.md](../Specifications/Notebooks/MilaCommandLineReview.md).
