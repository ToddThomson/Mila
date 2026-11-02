#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstddef>

namespace Mila::Dnn::Compute::Cuda
{
    // ============================================================================
    // Tensor Fill Operations (TensorFill.cu)
    // ============================================================================

    /**
     * @brief Templated fill operations with compile-time type dispatch
     */
    template<typename TargetType, typename HostType>
    void launch_constant_fill_typed( void* dst, size_t count, HostType host_value, cudaStream_t stream );

    template<typename TargetType, typename HostType>
    void launch_array_fill_typed( void* dst, const HostType* host_values, size_t count, cudaStream_t stream );
}