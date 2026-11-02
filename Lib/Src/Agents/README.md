# Mila::Agents

The agents API of the Mila library: a model at work in your program, with its conversation, its tools, and the
loop that runs their calls. The models API runs a model; the agents API puts one to work.

```cpp
import Mila;
import Mila.Agents;

auto agent = Mila::Agents::Agent::create( "gemma-4-12b-it-fp4" );
const Mila::Agents::Response response = agent.respond( "What is a KV cache?" );
```

An agent holds a `Mila::Agents::Session`: what your program gave it and the conversation. The session outlives the model, so
a program changes model or context and keeps its tools, rules and conversation.

Every call a model makes passes one gate: its arguments are checked, your rules allow it, ask the person or refuse it,
and it runs under a grant that holds it to the folders and network it was given.

It provides tools for files (`FileTools::tools()`) and web pages (`WebPageReader::tool()`), each with what the model
is told and what the person is shown; you choose which to offer, and any other tool is yours.
The grant's checks are Windows first: on any other platform a grant holds no folder.

## Using it

It is on by default, as the target `Mila::Agents`. A program that embeds Mila through `FetchContent` links it:

```cmake
FetchContent_MakeAvailable( Mila )
target_link_libraries( your-program PRIVATE Mila::Agents )
```

Running a tool's programs in Microsoft's MXC container is experimental and off by default: `MILA_ENABLE_MXC`, with
`MILA_MXC_LIBRARY` naming its library.

The design is in [Direction.md](../../../Specifications/Direction.md) sections 3.2 and 4 and
[ToolCalling.md](../../../Specifications/ToolCalling.md).
