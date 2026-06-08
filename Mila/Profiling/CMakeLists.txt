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
