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

# Define the Data/Datasets directory path. MILA_DATA_DIR is ${PROJECT_SOURCE_DIR}/Data,
# so this stays inside Mila's own tree when a consumer builds the samples as a subproject;
# CMAKE_SOURCE_DIR would be that consumer's root, which has no Datasets directory.
target_compile_definitions( MnistApp PRIVATE MILA_DATASETS_DIR="${MILA_DATA_DIR}/Datasets")

target_link_libraries( MnistApp PRIVATE Mila CUDA::cudart )
