# CMakeLists.txt for CharLM sample application

# Main CharLM training application
add_executable( CharLM
	"Src/CharLM.cpp"
)

set_target_properties( CharLM PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_sources( CharLM
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
		"Src/CharTransformer.ixx"
		"Src/CharDataLoader.ixx"
		
)

if( MILA_ENABLE_COPY_MODEL_DATA )
    add_dependencies( CharLM MilaCopyData )
endif()

target_link_libraries( CharLM PRIVATE Mila CUDA::cudart )

add_subdirectory( Tools )
