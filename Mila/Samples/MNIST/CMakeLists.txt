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

if(MILA_ENABLE_COPY_MODEL_DATA)
    add_dependencies(MnistApp MilaCopyData)
endif()

target_link_libraries( MnistApp PRIVATE Mila CUDA::cudart )
