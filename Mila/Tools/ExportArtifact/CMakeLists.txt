add_executable( ExportArtifact
    "ExportArtifact.cpp"
)

target_compile_features( ExportArtifact PRIVATE cxx_std_23 )

# Mila usage (model loading template instantiation) is confined to the module
# interface unit; ExportArtifact.cpp only imports it. The latest VS2026 MSVC raises
# C2079 (basic_istream::sentry undefined) when a plain .cpp consumer instantiates the
# readTensorBlob template, so that work must live in an .ixx. Same arrangement as
# Mila/Profiling/ProfileModel.
target_sources( ExportArtifact
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "ExportArtifact.ixx"
)

set_target_properties( ExportArtifact PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries( ExportArtifact PRIVATE
    Mila
)
