# CMakeList.txt : CMake project for Mila examples

# For Windows: Prevent overriding the parent project's compiler/linker settings
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

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

#set_target_properties( LayerNorm
#    PROPERTIES
#      _THRUST_HOST CPP
#      _THRUST_DEVICE CUDA
#)

target_link_libraries(LayerNorm PRIVATE Mila Thrust::Thrust CUDA::cudart )
