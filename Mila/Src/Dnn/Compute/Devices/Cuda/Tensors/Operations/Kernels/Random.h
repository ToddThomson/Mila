#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace Mila::Dnn::Compute::Cuda
{
    // Scales uniform [0, 1) values in-place to [min_val, max_val).
    void launch_scale_shift( float* data, size_t n, float min_val, float max_val, cudaStream_t stream );
}