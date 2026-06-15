# CMake project for Mila Samples

# MNIST is a CPU/CUDA MLP training sample, but as written it links CUDA::cudart
# and instantiates the CudaPinnedMemoryResource training path unconditionally, so
# it only compiles with CUDA enabled. Gate it the same way Chat is until the
# CPU-only build coherence work lands. Bard (GPT-2) revival is the next Training
# Revival slice and stays parked.
if(MILA_ENABLE_CUDA)
    add_subdirectory( MNIST )
else()
    message(STATUS "MILA_ENABLE_CUDA is OFF - skipping MNIST (CUDA-only build today)")
endif()
#add_subdirectory( Bard ) FIXME:Re-enable after alpha.5 completed

# Chat is an instruct, CUDA-targeted harness. CPU's only supported precision is
# FP32 and GPT-2 (the only CPU-practical model) is a base completion model, not
# an instruct model, so Chat has no coherent CPU build today. CPU inference is
# validated via tests and completion-oriented samples instead. Revisit if/when a
# CPU instruct path exists.
if(MILA_ENABLE_CUDA)
    add_subdirectory( Chat )
else()
    message(STATUS "MILA_ENABLE_CUDA is OFF - skipping Chat (instruct/CUDA-only sample)")
endif()

# QuickStart is intentionally NOT add_subdirectory()'d here. It is a standalone
# find_package(Mila) example/consumer that must be configured on its own against an
# install prefix, not built in-tree (an in-tree add_subdirectory would call
# find_package(Mila) before Mila is installed, and would link the in-tree target
# rather than the installed package). The Tests/Packaging gate drives it. See
# QuickStart/README.md.

if(WIN32)
  #add_subdirectory(Viz)
else()
  message(STATUS "Skipping Samples/Viz: Windows-only sample (not WIN32)")
endif()
