#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

    /**
     * @brief Generic value conversion between tensor data types
     *
     * Default implementation using static_cast for compatible types.
     */
    template <typename SrcT, typename DstT>
    __device__ DstT convert_value( SrcT val ) {
        return static_cast<DstT>(val);
    }

    /**
     * @brief Specialized conversion from FP16 to other types
     *
     * Converts __half to float first, then to target type for accuracy.
     */
    template <typename DstT>
    __device__ DstT convert_value( __half val ) {
        return static_cast<DstT>(__half2float( val ));
    }

    /**
     * @brief Specialized conversion from BFloat16 to other types
     *
     * Converts __nv_bfloat16 to float first, then to target type.
     */
    template <typename DstT>
    __device__ DstT convert_value( __nv_bfloat16 val ) {
        return static_cast<DstT>(__bfloat162float( val ));
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

    template <typename SrcT, typename DstT>
    void launch_convert_copy_kernel( const SrcT* d_src, DstT* d_or_h_dst, size_t n, cudaStream_t stream ) {
        if ( n == 0 ) return;
        constexpr int block = 256;
        int grid = static_cast<int>((n + block - 1) / block);
        
        convert_copy_kernel<SrcT, DstT> <<<grid, block, 0, stream>>> (d_src, d_or_h_dst, n);
    }

    template <typename T>
    void launch_fast_copy_kernel( const T* d_src, T* d_dst, size_t n, cudaStream_t stream ) {
        if ( n == 0 ) return;
        constexpr int block = 256;
        int grid = static_cast<int>((n + block - 1) / block);
        
        fast_copy_kernel<T> << <grid, block, 0, stream >> > (d_src, d_dst, n);
    }

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
        size_t n ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < n ) {
            dst[ idx ] = convert_value<DstT>( src[ idx ] );
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
        size_t dst_stride ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < n ) {
            size_t src_idx = idx * src_stride;
            size_t dst_idx = idx * dst_stride;
            dst[ dst_idx ] = convert_value<DstT>( src[ src_idx ] );
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
        size_t n ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Use vectorized loads/stores for supported types
        if constexpr ( sizeof( T ) == 4 ) {
            // 4-byte types (float, int32_t) - use float4 vectorization
            size_t vec_idx = idx * 4;
            if ( vec_idx + 3 < n ) {
                float4 data = reinterpret_cast<const float4*>(src)[ idx ];
                reinterpret_cast<float4*>(dst)[ idx ] = data;
                return;
            }
        }

        // Fallback to scalar copy for boundary elements or unsupported types
        if ( idx < n ) {
            dst[ idx ] = src[ idx ];
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
    __global__ void fill_kernel( T* __restrict__ dst, T value, size_t n ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx < n ) {
            dst[ idx ] = value;
        }
    }