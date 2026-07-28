#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace Mila::Dnn::Compute::Cuda
{
    // Scales uniform [0, 1) values in-place to [min_val, max_val).
    void launch_scale_shift( float* data, size_t n, float min_val, float max_val, cudaStream_t stream );

    // Narrowing a generated FP32 buffer into a BF16/FP16 tensor is done with
    // launch_convert_copy_kernel from Kernels/Transfer.Copy.h -- the same primitive the
    // transfer path uses. There is deliberately no random-specific converter here.
}
