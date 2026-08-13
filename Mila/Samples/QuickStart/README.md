# Quick Start

The shortest path to a running Mila, one directory per way in. Pick the one that matches how you
want to use Mila.

Both do the same thing — one prompt in, generated tokens streamed out, same model, same
template, same defaults — so you can read them side by side and see only the language differ.

| | Who it's for | What it costs |
|---|---|---|
| **[Python](Python/quickstart.py)** | I'm building in Python | `pip install mila-llm` and a model download. No compiler. Python 3.12 or 3.13, to match the published wheels. |
| **[C++](Cpp/)** | I'm building a C++ app | A C++23 toolchain, CUDA, and a from-source build of Mila. Its `CMakeLists.txt` is also the worked example of depending on Mila. |

Both need the same two things underneath: **a CUDA GPU**, and **a model in the local store**.
Models are not in git and nothing downloads one behind your back — installing is an explicit
step, and pull and load are separate verbs.

**Single-shot on purpose.** Neither has a conversation loop: history and a REPL teach nothing
about Mila and are most of what makes a chat harness large. `Mila/Adaptors/Chat` is where
multi-turn, channel routing and tool calls live — it is the payoff for going deeper, not the
thing to read first.

## Getting a model

The store is shared: Chat, the inference server, and the Python binding all read the same place,
so a model installed once is loadable by everything.

From Python:

```python
import mila
mila.initialize("warning")
mila.ModelStore().pull("gemma-4-12b-it-fp4", mila.default_hub_owner())
```

Or from the chat harness — `/models --online` to see what is published, `/install <name>` to take
one. Published today: `gemma-4-12b-it-fp4` (~6.3 GB), `Llama-3.2-3B-Instruct-fp4`,
`Llama-3.1-8B-Instruct-fp4`, and `gpt2-small`.

## After the quick start

`Python/chat.py` is the multi-turn version — streaming with the reasoning channels filtered out.
`Python/generate.py` covers the sampling knobs, `Python/store.py` the model store. In C++, the
chat harness at `Mila/Adaptors/Chat` is the complete article: streaming channels, tool calls and
model switching.

`Samples/Bard` and `Samples/MNIST` are the training samples — a full forward, backward and AdamW
loop, which is the other half of what Mila does.
[getting-started.md](../../../getting-started.md) is the long-form version of everything here,
including building Mila itself from a clone.
