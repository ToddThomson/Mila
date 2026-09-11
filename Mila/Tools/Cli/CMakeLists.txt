# The `mila` command. Named in lower case because it is what a user types, not a target
# style: every published quick start says `mila install <name>`.
add_executable( mila
    "main.cpp"
)

target_compile_features( mila PRIVATE cxx_std_23 )

# Mila usage is confined to the module interface unit; main.cpp only imports it. Same
# arrangement as ExportArtifact and ProfileModel -- MSVC raises C2079 when a plain .cpp
# consumer instantiates Mila templates directly.
target_sources( mila
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "Cli.ixx"
)

set_target_properties( mila PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries( mila PRIVATE
    Mila
)
