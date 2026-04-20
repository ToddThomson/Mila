add_executable( ProfilePrefill
    "ProfilePrefill.cpp"
 )

set_target_properties( ProfilePrefill PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

set( MODELS_DIR "${CMAKE_SOURCE_DIR}/Data/Models" )
target_compile_definitions( ProfilePrefill PRIVATE MODELS_DIR="${MODELS_DIR}" )

#target_link_libraries( ProfilePrefill )