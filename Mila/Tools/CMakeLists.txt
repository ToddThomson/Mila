# Mila Tools CMakeLists.txt
add_subdirectory( Tokenize )

# Artifact export needs a built, weights-loaded model to read quantized weights off
# the device, so it is CUDA-only.
if(MILA_ENABLE_CUDA)
    add_subdirectory( ExportArtifact )
endif()

