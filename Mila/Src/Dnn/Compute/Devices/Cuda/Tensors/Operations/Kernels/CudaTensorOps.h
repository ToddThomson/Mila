/**
 * @file CudaTensorOps.h
 * @brief CUDA tensor operation kernel function declarations
 * 
 * This header provides C-style function declarations for CUDA kernel wrappers
 * used by CUDA tensor operations. All functions are implemented in corresponding
 * .cu files and linked during compilation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace Mila::Dnn::Compute::Cuda
{
    // ============================================================================
    // Future Extension Points
    // ============================================================================

    // Additional kernel declarations can be added here as new .cu files are created:
    // - Random number generation kernels (cuRAND integration)
    // - Mathematical operation kernels (GELU, activation functions)
    // - Reduction kernels (sum, mean, max, min)
    // - Transformation kernels (transpose, reshape, permute)
}