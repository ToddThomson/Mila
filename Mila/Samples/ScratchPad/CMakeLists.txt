# CMakeList.txt : CMake project for Mila examples

# Add executable
add_executable(ScratchPad "main.cpp")

set_target_properties( ScratchPad
    PROPERTIES
      _THRUST_HOST CPP
      _THRUST_DEVICE CUDA
)

target_link_libraries(ScratchPad PRIVATE Mila Thrust::Thrust CUDA::cudart )
