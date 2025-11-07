# PreprocessText utility tool
add_executable( PreprocessText
	"PreprocessText.cpp"
)

set_target_properties( PreprocessText PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Tools"
)

target_sources( PreprocessText
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
		"CharVocabulary.ixx"
		"CharPreprocessor.ixx"
)

target_link_libraries( PreprocessText PRIVATE Mila )