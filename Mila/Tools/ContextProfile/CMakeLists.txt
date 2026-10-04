add_executable( ContextProfile
    "ContextProfile.cpp"
)

target_compile_features( ContextProfile PRIVATE cxx_std_23 )

# Every network instantiation lives in the module interface unit; ContextProfile.cpp only imports it.
# MSVC raises C2079 (basic_istream::sentry undefined) when a plain .cpp that imports Mila instantiates the
# weights reader, so that work must live in an .ixx -- the arrangement ExportArtifact and ProfileModel use.
target_sources( ContextProfile
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "ContextProfile.ixx"
)

set_target_properties( ContextProfile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# MILA_DATA_DIR, not CMAKE_SOURCE_DIR: the latter is the consumer's root in a subproject build.
target_compile_definitions( ContextProfile PRIVATE MILA_DATA_DIR="${MILA_DATA_DIR}" )

target_link_libraries( ContextProfile PRIVATE
    Mila
    MilaMeasurement
)

# Seven transformer instantiations in one translation unit cross MSVC's 65535-section object limit
# (C1128) under RelWithDebInfo, as ExportArtifact and ProfileModel do. /bigobj widens the COFF section
# count; it changes no code generation. MSVC-only: clang and gcc have no such limit.
if(MSVC)
    target_compile_options( ContextProfile PRIVATE /bigobj )
endif()
