add_executable( mila-metal-report
    "main.cpp"
    "GpuCoreCount.cpp" )

find_library( IOKIT_FRAMEWORK IOKit )
find_library( COREFOUNDATION_FRAMEWORK CoreFoundation )

target_link_libraries( mila-metal-report PRIVATE
    Mila
    ${IOKIT_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK} )
