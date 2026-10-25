# Mila.AI (Specifications/Direction.md 3.2): the layer applications are built on, a peer of the library that the
# library never imports. It starts in v0.21.0 with the tool broker's client as its only content, which needs a pipe
# and JSON and nothing of the library (ToolBroker.md section 8), so it links nothing of Mila yet.
add_library( MilaAI STATIC )
add_library( Mila::AI ALIAS MilaAI )

# The pipe is Windows' alone this release, as the broker is; its body is an implementation unit chosen here.
target_sources( MilaAI
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "AI.Json.ixx"
        "Mila.AI.ixx"
        "Tools/AI.Tools.ApprovalChoice.ixx"
        "Tools/AI.Tools.ApprovalRequest.ixx"
        "Tools/AI.Tools.BrokerClient.ixx"
        "Tools/AI.Tools.BrokerConnection.ixx"
        "Tools/AI.Tools.BrokerTool.ixx"
        "Tools/AI.Tools.ToolResult.ixx"
PRIVATE
    "Tools/BrokerClient.cpp"
    "Tools/BrokerConnection.Windows.cpp"
)

set_target_properties( MilaAI PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# The client's version in MCP's client information is Mila's own.
target_compile_definitions( MilaAI PRIVATE MILA_AI_VERSION="${PROJECT_VERSION}" )

target_link_libraries( MilaAI PUBLIC MilaStd )
