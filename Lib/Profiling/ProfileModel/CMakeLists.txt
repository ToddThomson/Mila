add_executable( ProfileModel
    "ProfileModel.cpp"
)

target_compile_features( ProfileModel PRIVATE cxx_std_23 )

# Mila usage (model loading template instantiation) is confined to the module
# interface unit; main.cpp only imports it. The latest VS2026 MSVC raises C2079
# (basic_istream::sentry undefined) when a plain .cpp consumer instantiates the
# readTensorBlob/seekg template, so that work must live in an .ixx.
target_sources( ProfileModel
PUBLIC
    FILE_SET module_files
    TYPE CXX_MODULES FILES
        "ProfileModel.ixx"
)

set_target_properties( ProfileModel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# MILA_DATA_DIR, not CMAKE_SOURCE_DIR: the latter is the consumer's root in a
# subproject build, and Mila's models are not under it.
set( MODELS_DIR "${MILA_DATA_DIR}/Models" )
target_compile_definitions( ProfileModel PRIVATE MODELS_DIR="${MODELS_DIR}" )

# ProfileModel.ixx loads every family at every quantization in one translation unit, and
# Gemma instantiates a dense and a routed (mixture-of-experts) transformer for each, which
# crosses MSVC's 65535-section object limit (C1128). /bigobj widens the COFF section count;
# it changes no code generation. MSVC-only: clang and gcc have no such limit.
if(MSVC)
    target_compile_options( ProfileModel PRIVATE /bigobj )
endif()

# Mila provides the model/tokenizer modules and the Profiling.NvtxRange module;
# CUDA::cudart provides cudaProfilerStart/Stop (cuda_profiler_api.h);
# CUDA::nvtx3 backs the NVTX ranges.
target_link_libraries( ProfileModel PRIVATE
    Mila
    CUDA::cudart
    CUDA::nvtx3
)
