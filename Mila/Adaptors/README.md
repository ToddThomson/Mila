# Adaptors

Consumer-facing adaptors over the Mila runtime library. The runtime generates
tokens and does not know or care who reads them; each adaptor here bridges those
tokens to a specific consumer, distinguished by **who closes the generation loop**:

- **[Chat](Chat/)** — closes the loop in-process with a **human** in the gate. An
  instruct chat CLI (`ConsoleRenderer`, channel-aware streaming, tool calling); the
  default model is Gemma 4 12B Instruct at FP4. CUDA-only today.
- **[Inference](Inference/)** — the **Mila Inference Server (MIS)**: exports the loop
  over an OpenAI/Anthropic-compatible wire so a foreign harness (Codex, Claude Code)
  drives Mila from another process. `Inference/Server` is the Python server; it imports
  the `mila` extension built from the runtime-adjacent [Mila/Bindings](../Bindings/)
  surface (not part of this adaptor). Consumer is a **machine**.
- **Agentic** (future) — closes the loop on **itself** under an autonomy policy: no
  external consumer, unsupervised, on-device.

These are first-class product surfaces, not demos (throwaway samples live under
`Mila/Samples`). The full positioning — runtime vs. adaptors, the layering, and the
shared native agent core Chat and Agentic will grow into — is in
[MilaProductFamily.md](../Specifications/MilaProductFamily.md).
