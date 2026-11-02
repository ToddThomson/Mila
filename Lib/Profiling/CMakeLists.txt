# CMake project for Mila Profiling

# ProfileModel is intrinsically a CUDA profiling harness: it calls
# cudaProfilerStart/Stop (cuda_profiler_api.h) and emits NVTX ranges, linking
# CUDA::cudart and CUDA::nvtx3. Those imported targets only exist when CUDA is
# enabled, and there is nothing to profile in a CPU-only build, so skip it.
if(MILA_ENABLE_CUDA)
    add_subdirectory( ProfileModel )
else()
    message(STATUS "MILA_ENABLE_CUDA is OFF - skipping ProfileModel (CUDA-only profiling harness)")
endif()

# Microbenchmarks/ is deliberately NOT added here. Each file is one translation unit
# built by hand with nvcc against architecture flags no preset carries -- notably
# -gencode=arch=compute_120f,code=sm_120f, since bare sm_120 silently drops the family
# suffix and ptxas then rejects every block-scaled instruction. They are run when a
# question is asked, not on every build. See Microbenchmarks/README.md.
