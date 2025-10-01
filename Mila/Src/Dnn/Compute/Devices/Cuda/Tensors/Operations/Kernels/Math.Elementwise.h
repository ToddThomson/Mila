/**
 * @file Math.Elementwise.h
 * @brief CUDA kernel declarations for element-wise tensor mathematical operations
 *
 * Provides launch function declarations for optimized CUDA kernels that perform
 * element-wise operations on tensors including binary arithmetic, scalar operations,
 * and comparison functions. All kernels are optimized for coalesced memory access
 * and support various data types through template specialization.
 */

#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace Mila::Dnn::Compute::Cuda::Kernels
{
    // ================================================================
    // Element-wise Binary Operations
    // ================================================================

    /**
     * @brief Launch element-wise tensor addition: dst = src1 + src2
     * @param src1 First source tensor data
     * @param src2 Second source tensor data
     * @param dst Destination tensor data
     * @param n Number of elements to process
     * @param stream CUDA stream for kernel execution
     */
    template<typename T>
    void launch_elementwise_add_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch element-wise tensor subtraction: dst = src1 - src2
     */
    template<typename T>
    void launch_elementwise_subtract_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch element-wise tensor multiplication: dst = src1 * src2
     */
    template<typename T>
    void launch_elementwise_multiply_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch element-wise tensor division: dst = src1 / src2
     */
    template<typename T>
    void launch_elementwise_divide_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch element-wise maximum: dst = max(src1, src2)
     */
    template<typename T>
    void launch_elementwise_max_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch element-wise minimum: dst = min(src1, src2)
     */
    template<typename T>
    void launch_elementwise_min_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    // ================================================================
    // Scalar Operations
    // ================================================================

    /**
     * @brief Launch scalar addition: dst = src + scalar
     * @param src Source tensor data
     * @param dst Destination tensor data
     * @param scalar Scalar value to add
     * @param n Number of elements to process
     * @param stream CUDA stream for kernel execution
     */
    template<typename T>
    void launch_scalar_add_kernel( const T* src, T* dst, T scalar,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch scalar multiplication: dst = src * scalar
     */
    template<typename T>
    void launch_scalar_multiply_kernel( const T* src, T* dst, T scalar,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch scalar division: dst = src / scalar
     */
    template<typename T>
    void launch_scalar_divide_kernel( const T* src, T* dst, T scalar,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch scalar subtraction: dst = src - scalar
     */
    template<typename T>
    void launch_scalar_subtract_kernel( const T* src, T* dst, T scalar,
        size_t n, cudaStream_t stream );

    // ================================================================
    // Comparison Operations
    // ================================================================

    /**
     * @brief Launch element-wise equality: dst = (src1 == src2) ? 1 : 0
     */
    template<typename T>
    void launch_elementwise_equal_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch element-wise greater than: dst = (src1 > src2) ? 1 : 0
     */
    template<typename T>
    void launch_elementwise_greater_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    /**
     * @brief Launch element-wise less than: dst = (src1 < src2) ? 1 : 0
     */
    template<typename T>
    void launch_elementwise_less_kernel( const T* src1, const T* src2, T* dst,
        size_t n, cudaStream_t stream );

    // ================================================================
    // Unary Operations
    // ================================================================

    /**
     * @brief Launch absolute value: dst = abs(src)
     */
    template<typename T>
    void launch_abs_kernel( const T* src, T* dst, size_t n, cudaStream_t stream );

    /**
     * @brief Launch negation: dst = -src
     */
    template<typename T>
    void launch_negate_kernel( const T* src, T* dst, size_t n, cudaStream_t stream );

    /**
     * @brief Launch square: dst = src * src
     */
    template<typename T>
    void launch_square_kernel( const T* src, T* dst, size_t n, cudaStream_t stream );

    /**
     * @brief Launch square root: dst = sqrt(src)
     */
    template<typename T>
    void launch_sqrt_kernel( const T* src, T* dst, size_t n, cudaStream_t stream );
}