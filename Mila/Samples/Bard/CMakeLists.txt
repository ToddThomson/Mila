# CMakeLists.txt for the Bard sample application

# Main Bard sample application
add_executable( Bard
	"Src/Bard.cpp"
)

set_target_properties( Bard PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( Bard
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
		"Src/BardTrainer.ixx"
		"Src/BardConfig.ixx"
)

# Absolute source-tree dataset path, mirroring MNIST, so the sample finds its
# Shakespeare data regardless of the working directory it is launched from.
target_compile_definitions( Bard PRIVATE MILA_DATASETS_DIR="${CMAKE_SOURCE_DIR}/Data/Datasets" )

target_link_libraries( Bard PRIVATE Mila CUDA::cudart )