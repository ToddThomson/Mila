add_executable( mila-metal-matrix-timing
    "main.cpp" )

# Raw metal-cpp and no Mila, so that an arm is the Metal calls alone.
target_include_directories( mila-metal-matrix-timing PRIVATE "${metal_cpp_SOURCE_DIR}" )

target_link_libraries( mila-metal-matrix-timing PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK} )
