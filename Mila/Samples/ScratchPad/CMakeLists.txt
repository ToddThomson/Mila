# CMakeList.txt : CMake project for Mila examples

# Add executable
add_executable(ScratchPad "main.cpp")
target_link_libraries(ScratchPad PUBLIC Mila Thrust::Thrust )
