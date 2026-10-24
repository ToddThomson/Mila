# The tool broker (Specifications/ToolBroker.md). Its name and command are not chosen yet (section 10.3), so only
# its library is built here; the tests link it, and the executable follows once it has a name.
add_library( MilaToolBroker STATIC )

# The Windows bodies are module implementation units chosen here, so the interfaces carry no platform
# preprocessor and no Windows header. Windows is the only platform this release (section 6).
target_sources( MilaToolBroker
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "ToolBroker.ClientMessage.ixx"
        "ToolBroker.Json.ixx"
        "ToolBroker.LineChannel.ixx"
        "ToolBroker.PipeServer.ixx"
        "ToolBroker.ProtocolError.ixx"
        "ToolBroker.Relay.ixx"
        "ToolBroker.Worker.ixx"
        "ToolBroker.WorkerLimits.ixx"
        "ToolBroker.WorkerProgram.ixx"
PRIVATE
    "LineChannel.Windows.cpp"
    "PipeServer.Windows.cpp"
    "Worker.Windows.cpp"
)

set_target_properties( MilaToolBroker PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Nothing of the library: the broker needs a pipe, processes and JSON, never a model (section 8). JSON is the
# vendored header Mila itself wraps (ToolBroker.Json.ixx).
target_link_libraries( MilaToolBroker PUBLIC MilaStd )
