# Mila Tools CMakeLists.txt
add_subdirectory( Tokenize )

# The `mila` command. Needs only the store and the hub, so it builds on any backend --
# unlike ExportArtifact below, which reads quantized weights off a device.
add_subdirectory( Cli )

# Artifact export needs a built, weights-loaded model to read quantized weights off
# the device, so it is CUDA-only.
if(MILA_ENABLE_CUDA)
    add_subdirectory( ExportArtifact )
endif()

