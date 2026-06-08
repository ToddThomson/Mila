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
		#"Src/BardTransformer.ixx"
		#"Src/BardTransformer.Config.ixx"
		#"Src/BardDataLoader.ixx"
		#"Tools/CharVocabulary.ixx"
		#"Src/BardTokenizer.ixx"
		"Src/BardTrainer.ixx"
		"Src/BardConfig.ixx"
)

target_compile_definitions( Bard PRIVATE MILA_DATA_DIR="${MILA_DATA_DIR}" )

target_link_libraries( Bard PRIVATE Mila CUDA::cudart )