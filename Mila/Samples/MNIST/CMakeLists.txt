add_executable( MnistApp 
	"Src/Mnist.cpp"
)

set_target_properties(MnistApp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( MnistApp
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
		"Src/MnistClassifier.ixx"
		"Src/MnistDataLoader.ixx"
		"Src/MnistConfig.ixx"
)

add_custom_command(
    TARGET MnistApp POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
        "${CMAKE_SOURCE_DIR}/Data/DataSets/Mnist/"
        "${CMAKE_BINARY_DIR}/Data/Mnist/"
    COMMENT "Copying data directory to build output directory"
)

target_link_libraries( MnistApp PRIVATE Mila CUDA::cudart )
