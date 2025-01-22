# CMakeList.txt : CMake project for Mila examples

# Add executable
add_executable( LayerNorm 
	"main.cpp"
)

target_sources( LayerNorm
PUBLIC
	FILE_SET module_files
       TYPE CXX_MODULES FILES
		"LayerNormModel.ixx" 
)

target_link_libraries(LayerNorm PUBLIC Mila Thrust::Thrust )
