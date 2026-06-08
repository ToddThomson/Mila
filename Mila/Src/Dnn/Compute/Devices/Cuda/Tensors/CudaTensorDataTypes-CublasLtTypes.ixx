/**
 * @file CudaTensorDataType-CublasLtTypes.ixx
 * @brief Compile-time mapping from abstract TensorDataType to cuBLASLt runtime cudaDataType_t enums.
 *
 * Provides CudaDataTypeTraits<TDataType>::cuda_data_type for use in cuBLASLt plan builders.
 * Complements TensorDataTypeMap (C++ device type) and lives as a partition of
 * Compute.CudaTensorDataType so it shares the module's established import chain.
 *
 * Only types that have a valid cudaDataType_t representation are specialised.
 * Instantiating the primary template for an unsupported type produces a clear static assertion.
 */

module;
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

export module Compute.CudaTensorDataType:CublasLtTypes;

import Dnn.TensorDataType;

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Compile-time mapping from TensorDataType -> cudaDataType_t
     *
     * Each specialisation exposes a single constexpr member:
     *   static constexpr cudaDataType_t cuda_data_type
     *
     * Used by cuBLASLt plan builders to select the correct layout data type
     * for matrix descriptors without threading runtime enums through call sites.
     *
     * @tparam TDataType  Abstract tensor data type (TensorDataType enum value)
     */
    export template<TensorDataType TDataType>
        struct CudaDataTypeTraits
    {
        static_assert(TDataType != TDataType, "No cudaDataType_t mapping for this TensorDataType");
    };

    // ====================================================================
    // Floating-Point Specialisations
    // ====================================================================

    template<>
    struct CudaDataTypeTraits<TensorDataType::FP32>
    {
        static constexpr cudaDataType_t cuda_data_type = CUDA_R_32F;
    };

    template<>
    struct CudaDataTypeTraits<TensorDataType::FP16>
    {
        static constexpr cudaDataType_t cuda_data_type = CUDA_R_16F;
    };

    template<>
    struct CudaDataTypeTraits<TensorDataType::BF16>
    {
        static constexpr cudaDataType_t cuda_data_type = CUDA_R_16BF;
    };

    template<>
    struct CudaDataTypeTraits<TensorDataType::FP8_E4M3>
    {
        static constexpr cudaDataType_t cuda_data_type = CUDA_R_8F_E4M3;
    };

    template<>
    struct CudaDataTypeTraits<TensorDataType::FP8_E5M2>
    {
        static constexpr cudaDataType_t cuda_data_type = CUDA_R_8F_E5M2;
    };

    // ====================================================================
    // Integer Specialisations
    // ====================================================================

    template<>
    struct CudaDataTypeTraits<TensorDataType::INT8>
    {
        static constexpr cudaDataType_t cuda_data_type = CUDA_R_8I;
    };

    template<>
    struct CudaDataTypeTraits<TensorDataType::INT32>
    {
        static constexpr cudaDataType_t cuda_data_type = CUDA_R_32I;
    };

    // ====================================================================
    // Convenience Alias
    // ====================================================================

    /**
     * @brief Convenience alias for accessing the cudaDataType_t mapping directly.
     *
     * Usage:
     *   constexpr cudaDataType_t dt = cuda_data_type_v<TensorDataType::BF16>;
     *   // dt == CUDA_R_BF16
     */
    export template<TensorDataType TDataType>
        constexpr cudaDataType_t cuda_data_type_v = CudaDataTypeTraits<TDataType>::cuda_data_type;
}