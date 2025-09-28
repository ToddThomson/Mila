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

    // Forward declaration of TensorDataType enum
    enum class TensorDataType : int {
        FP32 = 0,
        FP16 = 1,
        BF16 = 2,
        FP8_E4M3 = 3,
        FP8_E5M2 = 4,
        INT8 = 5,
        INT16 = 6,
        INT32 = 7,
        UINT8 = 8,
        UINT16 = 9,
        UINT32 = 10
    };

    // ============================================================================
    // Tensor Fill Operations (TensorFill.cu)
    // ============================================================================


        /**
         * @brief Fill tensor with constant int32_t value using device-native kernels
         *
         * Launches optimized CUDA kernels to fill device tensor memory with a constant
         * value, performing quantization to the target tensor data type.
         *
         * @param dst Pointer to device tensor memory
         * @param count Number of elements to fill
         * @param host_value Constant int32_t value to fill with
         * @param target_type Target tensor data type for quantization
         * @param stream CUDA stream for asynchronous execution (0 for default stream)
         */
        void launch_fill_constant_int32_kernel( void* dst, size_t count, int32_t host_value,
            TensorDataType target_type, cudaStream_t stream );

        /**
         * @brief Fill tensor with constant float value using device-native kernels
         *
         * Launches optimized CUDA kernels to fill device tensor memory with a constant
         * value, performing quantization to the target tensor data type.
         *
         * @param dst Pointer to device tensor memory
         * @param count Number of elements to fill
         * @param host_value Constant float value to fill with
         * @param target_type Target tensor data type for quantization
         * @param stream CUDA stream for asynchronous execution (0 for default stream)
         */
        void launch_fill_constant_float_kernel( void* dst, size_t count, float host_value,
            TensorDataType target_type, cudaStream_t stream );

        /**
         * @brief Fill tensor from host int32_t array with chunked processing
         *
         * Transfers host int32_t values to device memory with quantization to target
         * data type. Automatically handles chunked processing for large arrays to
         * limit temporary memory usage.
         *
         * @param dst Pointer to device tensor memory
         * @param count Number of elements to transfer
         * @param host_values Pointer to host int32_t values
         * @param target_type Target tensor data type for quantization
         * @param stream CUDA stream for asynchronous execution (0 for default stream)
         */
        void launch_fill_int32_kernel( void* dst, size_t count, const int32_t* host_values,
            TensorDataType target_type, cudaStream_t stream );

        /**
         * @brief Fill tensor from host float array with chunked processing
         *
         * Transfers host float values to device memory with quantization to target
         * data type. Automatically handles chunked processing for large arrays to
         * limit temporary memory usage.
         *
         * @param dst Pointer to device tensor memory
         * @param count Number of elements to transfer
         * @param host_values Pointer to host float values
         * @param target_type Target tensor data type for quantization
         * @param stream CUDA stream for asynchronous execution (0 for default stream)
         */
        void launch_fill_float_kernel( void* dst, size_t count, const float* host_values,
            TensorDataType target_type, cudaStream_t stream );
    

    // ============================================================================
    // Tensor Copy Operations (TensorCopy.cu)
    // ============================================================================

        /**
         * @brief Launch type-converting copy kernel between tensors
         *
         * Copies data between tensors with automatic type conversion using
         * specialized device conversion functions for optimal performance.
         *
         * @tparam SrcT Source tensor element type
         * @tparam DstT Destination tensor element type
         * @param d_src Source device memory pointer
         * @param d_or_h_dst Destination memory pointer (device or mapped host)
         * @param n Number of elements to copy
         * @param stream CUDA stream for asynchronous execution (0 for default stream)
         */
        template <typename SrcT, typename DstT>
        void launch_convert_copy_kernel( const SrcT* d_src, DstT* d_or_h_dst, size_t n, cudaStream_t stream );

        /**
         * @brief Launch optimized same-type copy kernel
         *
         * Performs optimized copy between tensors of the same type using
         * vectorized memory operations when possible.
         *
         * @tparam T Tensor element type (same for source and destination)
         * @param d_src Source device memory pointer
         * @param d_dst Destination device memory pointer
         * @param n Number of elements to copy
         * @param stream CUDA stream for asynchronous execution (0 for default stream)
         */
        template <typename T>
        void launch_fast_copy_kernel( const T* d_src, T* d_dst, size_t n, cudaStream_t stream );

    // ============================================================================
    // Future Extension Points
    // ============================================================================

    // Additional kernel declarations can be added here as new .cu files are created:
    // - Random number generation kernels (cuRAND integration)
    // - Mathematical operation kernels (GELU, activation functions)
    // - Reduction kernels (sum, mean, max, min)
    // - Transformation kernels (transpose, reshape, permute)
}