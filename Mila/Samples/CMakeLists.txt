# CMake project for Mila Samples

# MNIST (MLP) and Bard (GPT-2) are CPU/CUDA training samples, but as written they
# link CUDA::cudart and instantiate the CudaPinnedMemoryResource training path
# unconditionally, so they only compile with CUDA enabled. Gate them the same way
# Chat is until the CPU-only build coherence work lands.
if(MILA_ENABLE_CUDA)
    add_subdirectory( MNIST )
    add_subdirectory( Bard )
else()
    message(STATUS "MILA_ENABLE_CUDA is OFF - skipping MNIST and Bard (CUDA-only build today)")
endif()

# Chat is no longer a sample: it graduated to a first-class adaptor under
# Mila/Adaptors/Chat (see MilaProductFamily.md). Its build wiring lives in
# Mila/Adaptors/CMakeLists.txt.

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
