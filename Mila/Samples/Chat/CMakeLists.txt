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

# Define the Data/Models directory path
set(MODELS_DIR "${CMAKE_SOURCE_DIR}/Data/Models")
target_compile_definitions( ChatApp PRIVATE MODELS_DIR="${MODELS_DIR}" )

target_link_libraries( ChatApp PRIVATE Mila CUDA::cudart )
