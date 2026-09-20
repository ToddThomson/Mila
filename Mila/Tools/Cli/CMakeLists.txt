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

# Not the build root: CMake's own subdirectory for Mila/ sits there, and on a case-insensitive
# filesystem -- a Windows bind mount into the dev container, or macOS -- `mila` and `Mila/` are
# the same name, so the link fails with "cannot open output file mila: Is a directory".
set_target_properties( mila PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries( mila PRIVATE
    Mila
)
