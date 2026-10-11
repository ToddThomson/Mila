# The `mila-store` command. Named in lower case with hyphens because it is what a user types,
# not a target style (Specifications/Notebooks/MilaCommandLineReview.md section 4.1).
add_executable( mila-store
    "main.cpp"
)

target_compile_features( mila-store PRIVATE cxx_std_23 )

# Mila usage is confined to the module interface unit; main.cpp only imports it. Same
# arrangement as ExportArtifact and ProfileModel -- MSVC raises C2079 when a plain .cpp
# consumer instantiates Mila templates directly.
target_sources( mila-store
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "Store.ixx"
)

# Beside mila-chat at the build root, so the commands a user runs sit in one directory.
set_target_properties( mila-store PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries( mila-store PRIVATE
    Mila
)
