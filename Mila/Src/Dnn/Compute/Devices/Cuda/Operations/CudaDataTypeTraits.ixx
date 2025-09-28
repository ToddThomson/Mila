module;
#include <type_traits>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

export module Cuda.DataTypeTraits;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Helper struct to map C++ types to CUDA data types for cuBLASLt
     * @tparam T The C++ data type
     */
    export template <typename T>
        struct CudaDataTypeMap {
        // Default case (should cause a compilation error if used with unsupported type)
        static_assert(std::is_void_v<T>, "Unsupported data type for CUDA computation");
    };

    // Specializations for supported types

    // Float (32-bit)
    export template <>
        struct CudaDataTypeMap<float> {
        static constexpr cudaDataType_t value = CUDA_R_32F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    };

    // Half precision (16-bit)
    export template <>
        struct CudaDataTypeMap<half> {
        static constexpr cudaDataType_t value = CUDA_R_16F;
        static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
        // Alternative compute types for mixed precision:
        static constexpr cublasComputeType_t fp32_compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    };

    // BFloat16 (16-bit)
    export template <>
        struct CudaDataTypeMap<__nv_bfloat16> {
        static constexpr cudaDataType_t value = CUDA_R_16BF;
        //static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_16BF;
        // Alternative compute types for mixed precision:
        static constexpr cublasComputeType_t fp32_compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
    };
}
