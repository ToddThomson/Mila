# First-class adaptors over the Mila runtime library. Each bridges generated
# tokens to a specific consumer (see Adaptors/README.md and MilaProductFamily.md):
# Chat -> a human at a prompt; MIS -> a foreign harness over the wire (Python server
# under Inference/Server; it imports the mila binding built from Mila/Bindings).

# Chat is an instruct, CUDA-targeted adaptor. CPU's only supported precision is
# FP32 and the only CPU-practical model (GPT-2) is a base completion model, not an
# instruct model, so Chat has no coherent CPU build today. Gated on top-level so a
# find_package/CPM consumer embedding Mila never builds the Chat application.
if(MILA_ENABLE_ADAPTORS AND MILA_ENABLE_CUDA)
    add_subdirectory( Chat )
elseif(MILA_ENABLE_ADAPTORS)
    message(STATUS "MILA_ENABLE_CUDA is OFF - skipping Chat adaptor (instruct/CUDA-only)")
endif()

# MIS is the wire adaptor: the Python inference server under Inference/Server. It is
# not a CMake target -- it consumes the Mila Python binding, which is a runtime-adjacent
# surface built from Mila/Bindings (added at the Mila/ level, not here), not an adaptor.
