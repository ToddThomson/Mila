/**
 * @file CudaTensorDataType-Maps.ixx
 * @brief CUDA-specific mappings between abstract `TensorDataType` and concrete CUDA native types
 *
 * This module provides compile-time mapping utilities used by CUDA tensor implementations:
 * - `TensorDataTypeMap<TensorDataType>`: Maps an abstract tensor data type to the
 *   concrete CUDA native type used on-device (for kernel / device code).
 *
 * The template is intentionally closed via explicit specializations. Instantiating
 * the primary template for an unsupported type produces a readable static assertion.
 *
 * Notes:
 * - This file depends on CUDA headers for native device types (half, bfloat16, FP8).
 * - Host type mappings are handled by the centralized TensorHostTypeMap module.
 */

module;
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

export module Compute.CudaTensorDataType:Maps;

import Dnn.TensorDataType;

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Compile-time mapping from abstract TensorDataType -> CUDA native device type
     *
     * Maps abstract tensor data types to their corresponding CUDA device-native types
     * for use in CUDA kernels and device code. Each specialization defines the concrete
     * CUDA type that implements the abstract tensor data type on NVIDIA GPUs.
     *
     * The primary template triggers a static assertion to provide a clear diagnostic for
     * unsupported mappings, ensuring that only explicitly supported types can be used.
     *
     * @tparam TDataType Abstract tensor data type to map
     *
     * @note This mapping is CUDA-specific and defines device-side native types only
     * @note Host type mappings are handled separately by TensorHostTypeMap
     * @note All specializations use CUDA-native types that require device compilation
     */
    export template<TensorDataType TDataType>
    struct TensorDataTypeMap {
        static_assert(TDataType != TDataType, "Unsupported CUDA data type mapping");
    };

    // ====================================================================
    // Floating-Point Type Specializations
    // ====================================================================

    /**
     * @brief Maps `TensorDataType::FP32` to CUDA `float`
     *
     * Standard 32-bit IEEE 754 floating point, natively supported by CUDA.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::FP32> {
        using native_type = float;
    };

    /**
     * @brief Maps `TensorDataType::FP16` to CUDA `__half`
     *
     * 16-bit half precision floating point using CUDA's native half type.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::FP16> {
        using native_type = __half;
    };

    /**
     * @brief Maps `TensorDataType::BF16` to CUDA `__nv_bfloat16`
     *
     * 16-bit brain floating point using NVIDIA's bfloat16 implementation.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::BF16> {
        using native_type = __nv_bfloat16;
    };

    /**
     * @brief Maps `TensorDataType::FP8_E4M3` to CUDA `__nv_fp8_e4m3`
     *
     * 8-bit floating point with 4-bit exponent and 3-bit mantissa using NVIDIA's FP8 format.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::FP8_E4M3> {
        using native_type = __nv_fp8_e4m3;
    };

    /**
     * @brief Maps `TensorDataType::FP8_E5M2` to CUDA `__nv_fp8_e5m2`
     *
     * 8-bit floating point with 5-bit exponent and 2-bit mantissa using NVIDIA's FP8 format.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::FP8_E5M2> {
        using native_type = __nv_fp8_e5m2;
    };

    // ====================================================================
    // Integer Type Specializations
    // ====================================================================

    /**
     * @brief Maps `TensorDataType::INT8` to `std::int8_t`
     *
     * 8-bit signed integer using standard C++ type.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::INT8> {
        using native_type = std::int8_t;
    };

    /**
     * @brief Maps `TensorDataType::INT16` to `std::int16_t`
     *
     * 16-bit signed integer using standard C++ type.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::INT16> {
        using native_type = std::int16_t;
    };

    /**
     * @brief Maps `TensorDataType::INT32` to `std::int32_t`
     *
     * 32-bit signed integer using standard C++ type.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::INT32> {
        using native_type = std::int32_t;
    };

    /**
     * @brief Maps `TensorDataType::UINT8` to `std::uint8_t`
     *
     * 8-bit unsigned integer using standard C++ type.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::UINT8> {
        using native_type = std::uint8_t;
    };

    /**
     * @brief Maps `TensorDataType::UINT16` to `std::uint16_t`
     *
     * 16-bit unsigned integer using standard C++ type.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::UINT16> {
        using native_type = std::uint16_t;
    };

    /**
     * @brief Maps `TensorDataType::UINT32` to `std::uint32_t`
     *
     * 32-bit unsigned integer using standard C++ type.
     */
    template<>
    struct TensorDataTypeMap<TensorDataType::UINT32> {
        using native_type = std::uint32_t;
    };

    // ====================================================================
    // Convenience Aliases
    // ====================================================================

    /**
     * @brief Convenience alias for accessing CUDA native type
     *
     * @tparam TDataType Abstract tensor data type
     */
    export template<TensorDataType TDataType>
        using native_type_t = typename TensorDataTypeMap<TDataType>::native_type;
}