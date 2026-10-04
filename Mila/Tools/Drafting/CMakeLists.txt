add_executable( Drafting
    "Drafting.cpp"
)

target_compile_features( Drafting PRIVATE cxx_std_23 )

# The network instantiation lives in the module interface unit; Drafting.cpp only imports it, as in ContextProfile.
target_sources( Drafting
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "Drafting.ixx"
)

set_target_properties( Drafting PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_compile_definitions( Drafting PRIVATE MILA_DATA_DIR="${MILA_DATA_DIR}" )

target_link_libraries( Drafting PRIVATE
    Mila
    MilaMeasurement
)

if(MSVC)
    target_compile_options( Drafting PRIVATE /bigobj )
endif()
