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

set( MODELS_DIR "${CMAKE_SOURCE_DIR}/Data/Models" )
target_compile_definitions( ProfileModel PRIVATE MODELS_DIR="${MODELS_DIR}" )

# Mila provides the model/tokenizer modules and the Profiling.NvtxRange module;
# CUDA::cudart provides cudaProfilerStart/Stop (cuda_profiler_api.h);
# CUDA::nvtx3 backs the NVTX ranges.
target_link_libraries( ProfileModel PRIVATE
    Mila
    CUDA::cudart
    CUDA::nvtx3
)
