add_executable( ChatApp 
    "Src/main.cpp"
)

set_target_properties( ChatApp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( ChatApp
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
        "Src/Chat.ixx"
	    "Src/Chat.Config.ixx"
)

target_link_libraries( ChatApp PRIVATE Mila CUDA::cudart )

add_custom_command(
    TARGET ChatApp POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
        "${CMAKE_SOURCE_DIR}/data"
        "${CMAKE_BINARY_DIR}/data"
    COMMENT "Copying data directory to build output directory"
)