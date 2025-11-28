#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <cstdint>

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Generic value conversion between tensor data types
     *
     * Default implementation using static_cast for compatible types.
     */
    template <typename SrcT, typename DstT>
    __device__ DstT convert_value( SrcT val )
    {
        return static_cast<DstT>(val);
    }

    /**
     * @brief Specialized conversion from FP16 to other types
     *
     * Converts __half to float first, then to target type for accuracy.
     */
    template <typename DstT>
    __device__ DstT convert_value( __half val )
    {
        return static_cast<DstT>(__half2float( val ));
    }

    /**
     * @brief Specialized conversion from BFloat16 to other types
     *
     * Converts __nv_bfloat16 to float first, then to target type.
     */
    template <typename DstT>
    __device__ DstT convert_value( __nv_bfloat16 val )
    {
        return static_cast<DstT>(__bfloat162float( val ));
    }

    /**
     * @brief Specialized conversion TO FP16 from other types
     *
     * Converts source type to float first, then to __half for accuracy.
     */
    template <typename SrcT>
    __device__ __half convert_value_to_half( SrcT val )
    {
        return __float2half( static_cast<float>(val) );
    }

    /**
     * @brief Specialized conversion TO BFloat16 from other types
     *
     * Converts source type to float first, then to __nv_bfloat16.
     */
    template <typename SrcT>
    __device__ __nv_bfloat16 convert_value_to_bfloat16( SrcT val )
    {
        return __float2bfloat16( static_cast<float>(val) );
    }

    /**
     * @brief Specialized conversion from FP8 E4M3 to other types
     *
     * Uses NVIDIA's native FP8 to FP32 conversion with saturation.
     */
     /*template <typename DstT>
     __device__ DstT convert_value( __nv_fp8_e4m3 val ) {
         return static_cast<DstT>(__nv_cvt_fp8_to_fp32( val, __NV_SATFINITE ));
     }*/

     /**
      * @brief Specialized conversion from FP8 E5M2 to other types
      *
      * Uses NVIDIA's native FP8 to FP32 conversion with saturation.
      */
      /*template <typename DstT>
      __device__ DstT convert_value( __nv_fp8_e5m2 val ) {
          return static_cast<DstT>(__nv_cvt_fp8_to_fp32( val, __NV_SATFINITE ));
      }*/

      /**
       * @brief CUDA kernel for tensor copy with type conversion
       *
       * Performs element-wise copy from source to destination tensor with
       * automatic type conversion using specialized convert_value functions.
       * Optimized for coalesced memory access patterns.
       *
       * @tparam SrcT Source tensor element type
       * @tparam DstT Destination tensor element type
       * @param src Source tensor data pointer
       * @param dst Destination tensor data pointer
       * @param n Number of elements to copy
       */
    template <typename SrcT, typename DstT>
    __global__ void convert_copy_kernel( const SrcT* __restrict__ src,
        DstT* __restrict__ dst,
        size_t n )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < n )
        {
            if constexpr ( std::is_same_v<DstT, __half> && !std::is_same_v<SrcT, __half> )
            {
                dst[idx] = convert_value_to_half( src[idx] );
            }
            else if constexpr ( std::is_same_v<DstT, __nv_bfloat16> && !std::is_same_v<SrcT, __nv_bfloat16> )
            {
                dst[idx] = convert_value_to_bfloat16( src[idx] );
            }
            else
            {
                dst[idx] = convert_value<SrcT, DstT>( src[idx] );
            }
        }
    }

    /**
     * @brief CUDA kernel for tensor copy with stride support
     *
     * Performs element-wise copy with support for non-contiguous memory
     * layouts using stride information for both source and destination.
     *
     * @tparam SrcT Source tensor element type
     * @tparam DstT Destination tensor element type
     * @param src Source tensor data pointer
     * @param dst Destination tensor data pointer
     * @param n Number of elements to copy
     * @param src_stride Stride for source tensor access
     * @param dst_stride Stride for destination tensor access
     */
    template <typename SrcT, typename DstT>
    __global__ void convert_copy_strided_kernel( const SrcT* __restrict__ src,
        DstT* __restrict__ dst,
        size_t n,
        size_t src_stride,
        size_t dst_stride )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < n )
        {
            size_t src_idx = idx * src_stride;
            size_t dst_idx = idx * dst_stride;
            if constexpr ( std::is_same_v<DstT, __half> && !std::is_same_v<SrcT, __half> )
            {
                dst[dst_idx] = convert_value_to_half( src[src_idx] );
            }
            else if constexpr ( std::is_same_v<DstT, __nv_bfloat16> && !std::is_same_v<SrcT, __nv_bfloat16> )
            {
                dst[dst_idx] = convert_value_to_bfloat16( src[src_idx] );
            }
            else
            {
                dst[dst_idx] = convert_value<SrcT, DstT>( src[src_idx] );
            }
        }
    }

    /**
     * @brief Optimized CUDA kernel for same-type tensor copy
     *
     * Specialized kernel for cases where source and destination types
     * are identical, enabling vectorized memory operations and higher
     * throughput for large tensor transfers.
     *
     * @tparam T Tensor element type (same for source and destination)
     * @param src Source tensor data pointer
     * @param dst Destination tensor data pointer
     * @param n Number of elements to copy
     */
    template <typename T>
    __global__ void fast_copy_kernel( const T* __restrict__ src,
        T* __restrict__ dst,
        size_t n )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Use vectorized loads/stores for supported types
        if constexpr ( sizeof( T ) == 4 )
        {
            // 4-byte types (float, int32_t) - use float4 vectorization
            size_t vec_idx = idx * 4;
            if ( vec_idx + 3 < n )
            {
                float4 data = reinterpret_cast<const float4*>(src)[idx];
                reinterpret_cast<float4*>(dst)[idx] = data;
                return;
            }
        }

        // Fallback to scalar copy for boundary elements or unsupported types
        if ( idx < n )
        {
            dst[idx] = src[idx];
        }
    }

    /**
     * @brief CUDA kernel for tensor fill operation
     *
     * Fills tensor with a constant value, useful for initialization
     * and zero operations.
     *
     * @tparam T Tensor element type
     * @param dst Destination tensor data pointer
     * @param value Value to fill tensor with
     * @param n Number of elements to fill
     */
    template <typename T>
    __global__ void fill_kernel( T* __restrict__ dst, T value, size_t n )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < n )
        {
            dst[idx] = value;
        }
    }

    template <typename SrcT, typename DstT>
    void launch_convert_copy_kernel( const SrcT* d_src, DstT* d_or_h_dst, size_t n, cudaStream_t stream )
    {
        if ( n == 0 ) return;
        constexpr int block = 256;
        int grid = static_cast<int>((n + block - 1) / block);

        convert_copy_kernel<SrcT, DstT> << <grid, block, 0, stream >> > (d_src, d_or_h_dst, n);
    }

    template <typename T>
    void launch_fast_copy_kernel( const T* d_src, T* d_dst, size_t n, cudaStream_t stream )
    {
        if ( n == 0 ) return;
        constexpr int block = 256;
        int grid = static_cast<int>((n + block - 1) / block);

        fast_copy_kernel<T> << <grid, block, 0, stream >> > (d_src, d_dst, n);
    }

    // ========================================================================
    // Explicit Template Instantiations
    // ========================================================================

    // Common float conversions
    template void launch_convert_copy_kernel<float, __half>( const float*, __half*, size_t, cudaStream_t );
    template void launch_convert_copy_kernel<__half, float>( const __half*, float*, size_t, cudaStream_t );
    template void launch_convert_copy_kernel<float, __nv_bfloat16>( const float*, __nv_bfloat16*, size_t, cudaStream_t );
    template void launch_convert_copy_kernel<__nv_bfloat16, float>( const __nv_bfloat16*, float*, size_t, cudaStream_t );

    // Half precision conversions
    template void launch_convert_copy_kernel<__half, __nv_bfloat16>( const __half*, __nv_bfloat16*, size_t, cudaStream_t );
    template void launch_convert_copy_kernel<__nv_bfloat16, __half>( const __nv_bfloat16*, __half*, size_t, cudaStream_t );

    // Integer conversions
    template void launch_convert_copy_kernel<float, int32_t>( const float*, int32_t*, size_t, cudaStream_t );
    template void launch_convert_copy_kernel<int32_t, float>( const int32_t*, float*, size_t, cudaStream_t );
    template void launch_convert_copy_kernel<__half, int32_t>( const __half*, int32_t*, size_t, cudaStream_t );
    template void launch_convert_copy_kernel<int32_t, __half>( const int32_t*, __half*, size_t, cudaStream_t );

    // Fast copy kernels (same type)
    template void launch_fast_copy_kernel<float>( const float*, float*, size_t, cudaStream_t );
    template void launch_fast_copy_kernel<__half>( const __half*, __half*, size_t, cudaStream_t );
    template void launch_fast_copy_kernel<__nv_bfloat16>( const __nv_bfloat16*, __nv_bfloat16*, size_t, cudaStream_t );
    template void launch_fast_copy_kernel<int32_t>( const int32_t*, int32_t*, size_t, cudaStream_t );
    template void launch_fast_copy_kernel<int8_t>( const int8_t*, int8_t*, size_t, cudaStream_t );
    template void launch_fast_copy_kernel<uint8_t>( const uint8_t*, uint8_t*, size_t, cudaStream_t );
}