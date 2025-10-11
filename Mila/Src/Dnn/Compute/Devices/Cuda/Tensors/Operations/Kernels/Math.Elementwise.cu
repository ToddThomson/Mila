/**
 * @file Math.Elementwise.cu
 * @brief CUDA kernel implementations for element-wise tensor mathematical operations
 *
 * Optimized CUDA kernels for element-wise tensor operations with support for
 * coalesced memory access patterns, vectorized operations, and multiple data types.
 * All kernels use optimal block sizes and grid configurations for maximum throughput.
 */

#include "Math.Elementwise.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace Mila::Dnn::Compute::Cuda::Kernels
{
    // ================================================================
    // CUDA Kernel Implementations
    // ================================================================

    // Element-wise Binary Operations
    template<typename T>
    __global__ void elementwise_add_kernel(const T* __restrict__ src1, 
                                          const T* __restrict__ src2,
                                          T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src1[idx] + src2[idx];
        }
    }

    template<typename T>
    __global__ void elementwise_subtract_kernel(const T* __restrict__ src1,
                                               const T* __restrict__ src2,  
                                               T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src1[idx] - src2[idx];
        }
    }

    template<typename T>
    __global__ void elementwise_multiply_kernel(const T* __restrict__ src1,
                                               const T* __restrict__ src2,
                                               T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src1[idx] * src2[idx];
        }
    }

    template<typename T>
    __global__ void elementwise_divide_kernel(const T* __restrict__ src1,
                                             const T* __restrict__ src2,
                                             T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src1[idx] / src2[idx];
        }
    }

    template<typename T>
    __global__ void elementwise_max_kernel(const T* __restrict__ src1,
                                          const T* __restrict__ src2,
                                          T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = fmaxf(src1[idx], src2[idx]);
        }
    }

    template<typename T>
    __global__ void elementwise_min_kernel(const T* __restrict__ src1,
                                          const T* __restrict__ src2,
                                          T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = fminf(src1[idx], src2[idx]);
        }
    }

    // Scalar Operations
    template<typename T>
    __global__ void scalar_add_kernel(const T* __restrict__ src,
                                     T* __restrict__ dst, T scalar, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx] + scalar;
        }
    }

    template<typename T>
    __global__ void scalar_multiply_kernel(const T* __restrict__ src,
                                          T* __restrict__ dst, T scalar, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx] * scalar;
        }
    }

    template<typename T>
    __global__ void scalar_divide_kernel(const T* __restrict__ src,
                                        T* __restrict__ dst, T scalar, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx] / scalar;
        }
    }

    template<typename T>
    __global__ void scalar_subtract_kernel(const T* __restrict__ src,
                                          T* __restrict__ dst, T scalar, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx] - scalar;
        }
    }

    // Comparison Operations
    template<typename T>
    __global__ void elementwise_equal_kernel(const T* __restrict__ src1,
                                            const T* __restrict__ src2,
                                            T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = (src1[idx] == src2[idx]) ? T(1) : T(0);
        }
    }

    template<typename T>
    __global__ void elementwise_greater_kernel(const T* __restrict__ src1,
                                              const T* __restrict__ src2,
                                              T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = (src1[idx] > src2[idx]) ? T(1) : T(0);
        }
    }

    template<typename T>
    __global__ void elementwise_less_kernel(const T* __restrict__ src1,
                                           const T* __restrict__ src2,
                                           T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = (src1[idx] < src2[idx]) ? T(1) : T(0);
        }
    }

    // Unary Operations
    template<typename T>
    __global__ void abs_kernel(const T* __restrict__ src, T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = fabsf(src[idx]);
        }
    }

    template<typename T>
    __global__ void negate_kernel(const T* __restrict__ src, T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = -src[idx];
        }
    }

    template<typename T>
    __global__ void square_kernel(const T* __restrict__ src, T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = src[idx] * src[idx];
        }
    }

    template<typename T>
    __global__ void sqrt_kernel(const T* __restrict__ src, T* __restrict__ dst, size_t n) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            dst[idx] = sqrtf(src[idx]);
        }
    }

    // ================================================================
    // Launch Function Implementations
    // ================================================================

    template<typename T>
    void launch_elementwise_add_kernel(const T* src1, const T* src2, T* dst,
                                      size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_add_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_elementwise_subtract_kernel(const T* src1, const T* src2, T* dst,
                                           size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_subtract_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_elementwise_multiply_kernel(const T* src1, const T* src2, T* dst,
                                           size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_multiply_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_elementwise_divide_kernel(const T* src1, const T* src2, T* dst,
                                         size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_divide_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_elementwise_max_kernel(const T* src1, const T* src2, T* dst,
                                      size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_max_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_elementwise_min_kernel(const T* src1, const T* src2, T* dst,
                                      size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_min_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_scalar_add_kernel(const T* src, T* dst, T scalar,
                                 size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        scalar_add_kernel<T><<<grid, block, 0, stream>>>(src, dst, scalar, n);
    }

    template<typename T>
    void launch_scalar_multiply_kernel(const T* src, T* dst, T scalar,
                                      size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        scalar_multiply_kernel<T><<<grid, block, 0, stream>>>(src, dst, scalar, n);
    }

    template<typename T>
    void launch_scalar_divide_kernel(const T* src, T* dst, T scalar,
                                    size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        scalar_divide_kernel<T><<<grid, block, 0, stream>>>(src, dst, scalar, n);
    }

    template<typename T>
    void launch_scalar_subtract_kernel(const T* src, T* dst, T scalar,
                                      size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        scalar_subtract_kernel<T><<<grid, block, 0, stream>>>(src, dst, scalar, n);
    }

    template<typename T>
    void launch_elementwise_equal_kernel(const T* src1, const T* src2, T* dst,
                                        size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_equal_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_elementwise_greater_kernel(const T* src1, const T* src2, T* dst,
                                          size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_greater_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_elementwise_less_kernel(const T* src1, const T* src2, T* dst,
                                       size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        elementwise_less_kernel<T><<<grid, block, 0, stream>>>(src1, src2, dst, n);
    }

    template<typename T>
    void launch_abs_kernel(const T* src, T* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        abs_kernel<T><<<grid, block, 0, stream>>>(src, dst, n);
    }

    template<typename T>
    void launch_negate_kernel(const T* src, T* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        negate_kernel<T><<<grid, block, 0, stream>>>(src, dst, n);
    }

    template<typename T>
    void launch_square_kernel(const T* src, T* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        square_kernel<T><<<grid, block, 0, stream>>>(src, dst, n);
    }

    template<typename T>
    void launch_sqrt_kernel(const T* src, T* dst, size_t n, cudaStream_t stream) {
        if (n == 0) return;
        constexpr int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        sqrt_kernel<T><<<grid, block, 0, stream>>>(src, dst, n);
    }

    // ================================================================
// Explicit Template Instantiations
// ================================================================

// Addition kernels
    template void launch_elementwise_add_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_add_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_add_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_add_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    // Subtraction kernels
    template void launch_elementwise_subtract_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_subtract_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_subtract_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_subtract_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    // Multiplication kernels
    template void launch_elementwise_multiply_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_multiply_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_multiply_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_multiply_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    // Division kernels
    template void launch_elementwise_divide_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_divide_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_divide_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_divide_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    // Max/Min kernels
    template void launch_elementwise_max_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_max_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_max_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_max_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    template void launch_elementwise_min_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_min_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_min_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_min_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    // Scalar operation kernels
    template void launch_scalar_add_kernel<float>( const float*, float*, float, size_t, cudaStream_t );
    template void launch_scalar_add_kernel<double>( const double*, double*, double, size_t, cudaStream_t );
    template void launch_scalar_add_kernel<int>( const int*, int*, int, size_t, cudaStream_t );
    template void launch_scalar_add_kernel<__half>( const __half*, __half*, __half, size_t, cudaStream_t );

    template void launch_scalar_multiply_kernel<float>( const float*, float*, float, size_t, cudaStream_t );
    template void launch_scalar_multiply_kernel<double>( const double*, double*, double, size_t, cudaStream_t );
    template void launch_scalar_multiply_kernel<int>( const int*, int*, int, size_t, cudaStream_t );
    template void launch_scalar_multiply_kernel<__half>( const __half*, __half*, __half, size_t, cudaStream_t );

    template void launch_scalar_divide_kernel<float>( const float*, float*, float, size_t, cudaStream_t );
    template void launch_scalar_divide_kernel<double>( const double*, double*, double, size_t, cudaStream_t );
    template void launch_scalar_divide_kernel<int>( const int*, int*, int, size_t, cudaStream_t );
    template void launch_scalar_divide_kernel<__half>( const __half*, __half*, __half, size_t, cudaStream_t );

    template void launch_scalar_subtract_kernel<float>( const float*, float*, float, size_t, cudaStream_t );
    template void launch_scalar_subtract_kernel<double>( const double*, double*, double, size_t, cudaStream_t );
    template void launch_scalar_subtract_kernel<int>( const int*, int*, int, size_t, cudaStream_t );
    template void launch_scalar_subtract_kernel<__half>( const __half*, __half*, __half, size_t, cudaStream_t );

    // Comparison operation kernels
    template void launch_elementwise_equal_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_equal_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_equal_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_equal_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    template void launch_elementwise_greater_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_greater_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_greater_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_greater_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    template void launch_elementwise_less_kernel<float>( const float*, const float*, float*, size_t, cudaStream_t );
    template void launch_elementwise_less_kernel<double>( const double*, const double*, double*, size_t, cudaStream_t );
    template void launch_elementwise_less_kernel<int>( const int*, const int*, int*, size_t, cudaStream_t );
    template void launch_elementwise_less_kernel<__half>( const __half*, const __half*, __half*, size_t, cudaStream_t );

    // Unary operation kernels
    template void launch_abs_kernel<float>( const float*, float*, size_t, cudaStream_t );
    template void launch_abs_kernel<double>( const double*, double*, size_t, cudaStream_t );
    template void launch_abs_kernel<int>( const int*, int*, size_t, cudaStream_t );
    template void launch_abs_kernel<__half>( const __half*, __half*, size_t, cudaStream_t );

    template void launch_negate_kernel<float>( const float*, float*, size_t, cudaStream_t );
    template void launch_negate_kernel<double>( const double*, double*, size_t, cudaStream_t );
    template void launch_negate_kernel<int>( const int*, int*, size_t, cudaStream_t );
    template void launch_negate_kernel<__half>( const __half*, __half*, size_t, cudaStream_t );

    template void launch_square_kernel<float>( const float*, float*, size_t, cudaStream_t );
    template void launch_square_kernel<double>( const double*, double*, size_t, cudaStream_t );
    template void launch_square_kernel<int>( const int*, int*, size_t, cudaStream_t );
    template void launch_square_kernel<__half>( const __half*, __half*, size_t, cudaStream_t );

    template void launch_sqrt_kernel<float>( const float*, float*, size_t, cudaStream_t );
    template void launch_sqrt_kernel<double>( const double*, double*, size_t, cudaStream_t );
    template void launch_sqrt_kernel<int>( const int*, int*, size_t, cudaStream_t );
    template void launch_sqrt_kernel<__half>( const __half*, __half*, size_t, cudaStream_t );
}