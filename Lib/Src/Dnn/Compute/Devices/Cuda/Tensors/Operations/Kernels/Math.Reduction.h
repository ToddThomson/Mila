/**
 * @file Math.Reduction.h
 * @brief CUDA kernel declarations for tensor reduction operations (sum, mean, max, min)
 *
 * Declares launch functions for optimized CUDA reduction kernels. Each kernel writes
 * per-block partial results to a device buffer; a host-side final reduction is
 * expected by the caller (or a follow-up kernel).
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace Mila::Dnn::Compute::Cuda::Kernels
{
    /**
     * @brief Launch sum reduction kernel producing per-block partial sums (float)
     *
     * @tparam T Source element type
     * @param src Device pointer to source elements
     * @param d_partial_sums Device pointer to per-block partial sums (float)
     * @param count Number of elements in source
     * @param grid Number of blocks to launch (partial-sum entries)
     * @param block Threads per block
     * @param shared_bytes Shared memory size per block (bytes)
     * @param stream CUDA stream for kernel execution
     */
    template<typename T>
    void launch_sum_reduction_kernel(
        const T* src,
        float* d_partial_sums,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream );

    /**
     * @brief Launch mean reduction kernel producing per-block partial sums (float)
     *
     * Partial sums are written to the provided device buffer; caller computes final
     * mean by dividing the aggregated sum by the element count.
     */
    template<typename T>
    void launch_mean_reduction_kernel(
        const T* src,
        float* d_partial_means,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream );

    /**
     * @brief Launch max reduction kernel producing per-block partial maxima
     *
     * @tparam T Source and partial-result element type
     * @param src Device pointer to source elements
     * @param d_partial_maxes Device pointer to per-block partial maxima (T)
     * @param count Number of elements in source
     * @param grid Number of blocks to launch (partial-result entries)
     * @param block Threads per block
     * @param shared_bytes Shared memory size per block (bytes)
     * @param stream CUDA stream for kernel execution
     */
    template<typename T>
    void launch_max_reduction_kernel(
        const T* src,
        T* d_partial_maxes,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream );

    /**
     * @brief Launch min reduction kernel producing per-block partial minima
     *
     * @tparam T Source and partial-result element type
     */
    template<typename T>
    void launch_min_reduction_kernel(
        const T* src,
        T* d_partial_mins,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream );
}