/**
 * @file Math.Reduction.cu
 * @brief CUDA kernel implementations for tensor reduction operations (sum, mean, max, min)
 *
 * Implements per-block reduction kernels and launch wrappers declared in Math.Reduction.h.
 */

#include "Math.Reduction.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <limits>

namespace Mila::Dnn::Compute::Cuda::Kernels
{
    /*
     * Per-block sum reduction kernel.
     *
     * - Each block reduces a strided subset of the input into one float partial sum.
     * - Shared memory (float[]) must be provided with at least blockDim.x * sizeof(float).
     */
    template<typename T>
    __global__ static void sum_reduction_kernel(
        const T* src,
        float* d_partial_sums,
        size_t count,
        int grid,
        int block,
        size_t /*shared_bytes*/ )
    {
        extern __shared__ float sdata[];

        const unsigned int tid = threadIdx.x;
        const unsigned int block_threads = blockDim.x;
        const size_t global_start = static_cast<size_t>(blockIdx.x) * block_threads + tid;
        const size_t stride = static_cast<size_t>(grid) * block_threads;

        float local_sum = 0.0f;

        for ( size_t i = global_start; i < count; i += stride )
        {
            local_sum += static_cast<float>( src[i] );
        }

        sdata[tid] = local_sum;

        __syncthreads();

        // In-place reduction in shared memory
        for ( unsigned int s = block_threads / 2; s > 0; s >>= 1 )
        {
            if ( tid < s )
            {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads();
        }

        if ( tid == 0 )
        {
            d_partial_sums[blockIdx.x] = sdata[0];
        }
    }

    /*
     * Per-block mean reduction kernel (writes partial sums; final mean = aggregated_sum / count).
     *
     * Implementation identical to sum kernel; caller performs final division.
     */
    template<typename T>
    __global__ static void mean_reduction_kernel(
        const T* src,
        float* d_partial_means,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes )
    {
        sum_reduction_kernel<T>( src, d_partial_means, count, grid, block, shared_bytes );
    }

    /*
     * Per-block max reduction kernel.
     *
     * - Each block reduces a strided subset of the input into one partial maximum.
     * - Shared memory (T[]) must be provided with at least blockDim.x * sizeof(T).
     */
    template<typename T>
    __global__ static void max_reduction_kernel(
        const T* src,
        T* d_partial_maxes,
        size_t count,
        int grid,
        int block,
        size_t /*shared_bytes*/ )
    {
        extern __shared__ unsigned char raw[];
        T* sdata = reinterpret_cast<T*>(raw);

        const unsigned int tid = threadIdx.x;
        const unsigned int block_threads = blockDim.x;
        const size_t global_start = static_cast<size_t>(blockIdx.x) * block_threads + tid;
        const size_t stride = static_cast<size_t>(grid) * block_threads;

        // Initialize to smallest representable value for T
        T local_max = std::numeric_limits<T>::lowest();

        for ( size_t i = global_start; i < count; i += stride )
        {
            const T val = src[i];
            if ( val > local_max )
            {
                local_max = val;
            }
        }

        sdata[tid] = local_max;

        __syncthreads();

        for ( unsigned int s = block_threads / 2; s > 0; s >>= 1 )
        {
            if ( tid < s )
            {
                const T other = sdata[tid + s];
                if ( other > sdata[tid] )
                {
                    sdata[tid] = other;
                }
            }

            __syncthreads();
        }

        if ( tid == 0 )
        {
            d_partial_maxes[blockIdx.x] = sdata[0];
        }
    }

    /*
     * Per-block min reduction kernel.
     *
     * - Each block reduces a strided subset of the input into one partial minimum.
     */
    template<typename T>
    __global__ static void min_reduction_kernel(
        const T* src,
        T* d_partial_mins,
        size_t count,
        int grid,
        int block,
        size_t /*shared_bytes*/ )
    {
        extern __shared__ unsigned char raw[];
        T* sdata = reinterpret_cast<T*>(raw);

        const unsigned int tid = threadIdx.x;
        const unsigned int block_threads = blockDim.x;
        const size_t global_start = static_cast<size_t>(blockIdx.x) * block_threads + tid;
        const size_t stride = static_cast<size_t>(grid) * block_threads;

        T local_min = std::numeric_limits<T>::max();

        for ( size_t i = global_start; i < count; i += stride )
        {
            const T val = src[i];
            if ( val < local_min )
            {
                local_min = val;
            }
        }

        sdata[tid] = local_min;

        __syncthreads();

        for ( unsigned int s = block_threads / 2; s > 0; s >>= 1 )
        {
            if ( tid < s )
            {
                const T other = sdata[tid + s];
                if ( other < sdata[tid] )
                {
                    sdata[tid] = other;
                }
            }

            __syncthreads();
        }

        if ( tid == 0 )
        {
            d_partial_mins[blockIdx.x] = sdata[0];
        }
    }


    // -------------------------------
    // Launch wrappers
    // -------------------------------

    template<typename T>
    void launch_sum_reduction_kernel(
        const T* src,
        float* d_partial_sums,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream )
    {
        if ( count == 0 || grid <= 0 || block <= 0 )
        {
            return;
        }

        // Launch kernel
        sum_reduction_kernel<T> << < grid, block, shared_bytes, stream >> > (
            src, d_partial_sums, count, grid, block, shared_bytes
            );

        cudaError_t status = cudaGetLastError();
        if ( status != cudaSuccess )
        {
            throw std::runtime_error( std::string( "launch_sum_reduction_kernel failed: " ) +
                cudaGetErrorString( status ) );
        }
    }

    template<typename T>
    void launch_mean_reduction_kernel(
        const T* src,
        float* d_partial_means,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream )
    {
        if ( count == 0 || grid <= 0 || block <= 0 )
        {
            return;
        }

        mean_reduction_kernel<T> << < grid, block, shared_bytes, stream >> > (
            src, d_partial_means, count, grid, block, shared_bytes
            );

        cudaError_t status = cudaGetLastError();
        if ( status != cudaSuccess )
        {
            throw std::runtime_error( std::string( "launch_mean_reduction_kernel failed: " ) +
                cudaGetErrorString( status ) );
        }
    }

    template<typename T>
    void launch_max_reduction_kernel(
        const T* src,
        T* d_partial_maxes,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream )
    {
        if ( count == 0 || grid <= 0 || block <= 0 )
        {
            return;
        }

        max_reduction_kernel<T> << < grid, block, shared_bytes, stream >> > (
            src, d_partial_maxes, count, grid, block, shared_bytes
            );

        cudaError_t status = cudaGetLastError();
        if ( status != cudaSuccess )
        {
            throw std::runtime_error( std::string( "launch_max_reduction_kernel failed: " ) +
                cudaGetErrorString( status ) );
        }
    }

    template<typename T>
    void launch_min_reduction_kernel(
        const T* src,
        T* d_partial_mins,
        size_t count,
        int grid,
        int block,
        size_t shared_bytes,
        cudaStream_t stream )
    {
        if ( count == 0 || grid <= 0 || block <= 0 )
        {
            return;
        }

        min_reduction_kernel<T> << < grid, block, shared_bytes, stream >> > (
            src, d_partial_mins, count, grid, block, shared_bytes
            );

        cudaError_t status = cudaGetLastError();
        if ( status != cudaSuccess )
        {
            throw std::runtime_error( std::string( "launch_min_reduction_kernel failed: " ) +
                cudaGetErrorString( status ) );
        }
    }


    // -------------------------------
    // Explicit instantiations
    // -------------------------------
    //
    // Provide common instantiations here so the linker can find symbols for
    // commonly used types. Add more as required by project usage.
    //
    // Keep instantiations minimal; the project uses a mapping from abstract
    // TensorDataType to native CUDA types elsewhere. Instantiations below are
    // examples for standard arithmetic types.
    //

    // Sum / Mean instantiations (accumulated as float)
    template void launch_sum_reduction_kernel<float>(
        const float*, float*, size_t, int, int, size_t, cudaStream_t );

    template void launch_sum_reduction_kernel<double>(
        const double*, float*, size_t, int, int, size_t, cudaStream_t );

    template void launch_mean_reduction_kernel<float>(
        const float*, float*, size_t, int, int, size_t, cudaStream_t );

    template void launch_mean_reduction_kernel<double>(
        const double*, float*, size_t, int, int, size_t, cudaStream_t );

    // Max / Min instantiations
    template void launch_max_reduction_kernel<float>(
        const float*, float*, size_t, int, int, size_t, cudaStream_t );

    template void launch_min_reduction_kernel<float>(
        const float*, float*, size_t, int, int, size_t, cudaStream_t );

    template void launch_max_reduction_kernel<int32_t>(
        const int32_t*, int32_t*, size_t, int, int, size_t, cudaStream_t );

    template void launch_min_reduction_kernel<int32_t>(
        const int32_t*, int32_t*, size_t, int, int, size_t, cudaStream_t );

} // namespace Mila::Dnn::Compute::Cuda::Kernels