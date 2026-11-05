# Mila Tools CMakeLists.txt
add_subdirectory( Tokenize )

# What a Mac's GPU reports, and whether Mila's kernels compile on it (Specifications/Metal/MetalBackend.md).
if(MILA_HAS_METAL)
    add_subdirectory( MetalMatrixTiming )
    add_subdirectory( MetalReport )
endif()

# Artifact export needs a built, weights-loaded model to read quantized weights off
# the device, so it is CUDA-only.
if(MILA_ENABLE_CUDA)
    add_subdirectory( ExportArtifact )
endif()

# The context profile measures real models on a real card (Specifications/ContextProfile.md), so it is
# CUDA-only for the same reason.
if(MILA_ENABLE_CUDA)
    add_subdirectory( ContextProfile )
endif()

# Whether a draft model pays on a target (Specifications/Gemma4Mtp.md section 5.1), timed on a real card.
if(MILA_ENABLE_CUDA)
    add_subdirectory( Drafting )
endif()

