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

# Nothing under QuickStart/ is add_subdirectory()'d. It is the onboarding set -- one
# directory per path a newcomer can take to a first run -- and none of it builds here:
#
#   QuickStart/Cpp     a standalone FetchContent(Mila) project standing in for a
#                      downstream app, so it configures on its own and fetches Mila
#                      itself (an in-tree add_subdirectory would be circular). The
#                      Tests/Packaging packaging_fetchcontent_consumer gate validates
#                      the same mechanism against the local tree.
#   QuickStart/Python  scripts over the MilaPy extension, with nothing to build. They
#                      locate the extension at run time in the neutral output directory
#                      MilaPy publishes to (see Bindings/CMakeLists.txt).
#
# See QuickStart/README.md.

if(WIN32)
  #add_subdirectory(Viz)
else()
  message(STATUS "Skipping Samples/Viz: Windows-only sample (not WIN32)")
endif()
