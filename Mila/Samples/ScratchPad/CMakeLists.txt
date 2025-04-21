# CMakeList.txt : CMake project for Mila ScratchPad

# Add executable
add_executable(ScratchPad "main.cpp" "Linear.cpp" "LayerNorm.cpp"  "Softmax.cpp")

target_link_libraries(ScratchPad PRIVATE Mila CUDA::cudart )