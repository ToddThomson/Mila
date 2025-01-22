# CMakeList.txt : CMake project for Mila examples

# Add executable
add_executable( Gpt2App 
	"Src/TrainGpt2.cpp"
)

target_sources( Gpt2App
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
		"Src/ModelConfig.ixx"
		"Src/DataLoader.ixx"
		"Src/Gpt2Model.ixx"
		"Src/Tokenizer.ixx" 
)

target_link_libraries( Gpt2App PUBLIC Mila Thrust::Thrust )
#target_link_libraries( TrainGpt2 PRIVATE Mila Thrust::Thrust CUDA::nvml CUDA::cublas CUDA::cublasLt)  

add_custom_command(
    TARGET Gpt2App POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
        "${CMAKE_SOURCE_DIR}/data"
        "${CMAKE_BINARY_DIR}/data"
    COMMENT "Copying data directory to build output directory"
)

# Add Gpt2Test directory  
if (MILA_ENABLE_GPT2_TESTING)  
add_subdirectory(Tests)  
endif()  
