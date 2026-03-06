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

# Define the Data/Datasets directory path
target_compile_definitions( MnistApp PRIVATE MILA_DATASETS_DIR="${CMAKE_SOURCE_DIR}/Data/Datasets")

target_link_libraries( MnistApp PRIVATE Mila CUDA::cudart )
