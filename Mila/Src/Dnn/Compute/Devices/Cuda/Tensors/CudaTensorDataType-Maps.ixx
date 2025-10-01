/**
 * @file CudaTensorDataType.Maps.ixx
 * @brief CUDA-specific mappings between abstract `TensorDataType` and concrete types
 *
 * This module provides two small compile-time mapping utilities used by CUDA
 * tensor implementations:
 * - `CudaTensorDataTypeMap<TensorDataType>`: Maps an abstract tensor data type
 *   to the concrete CUDA native type used on-device (for kernel / device code).
 * - `HostDataTypeMap<TensorDataType>`: Maps an abstract tensor data type to the
 *   host-compatible type used when transferring or inspecting values on the host.
 *
 * Both templates are intentionally closed via explicit specializations. Instantiating
 * the primary template for an unsupported type produces a readable static assertion.
 *
 * Notes:
 * - This file depends on CUDA headers for native device types (half, bfloat16, FP8).
 * - Host mappings intentionally fold device-only small/fp types to `float` where
 *   appropriate to simplify host-side interactions and conversions.
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
     * Specialize this template for each TensorDataType that has a native CUDA representation.
     * The primary template triggers a static assertion to provide a clear diagnostic for
     * unsupported mappings.
     *
     * @tparam TDataType Abstract tensor data type to map
     */
    export template<TensorDataType TDataType>
        struct TensorDataTypeMap {
        static_assert(TDataType != TDataType, "Unsupported CUDA data type mapping (CudaTensorDataTypeMap)");
    };

    /** Concrete CUDA-native type specializations (device-side types) */

    /**
     * @brief Maps `TensorDataType::FP32` to CUDA `float`
     */
    template<> struct TensorDataTypeMap<TensorDataType::FP32> { using native_type = float; };

    /**
     * @brief Maps `TensorDataType::FP16` to CUDA `__half`
     */
    template<> struct TensorDataTypeMap<TensorDataType::FP16> { using type = __half; };

    /**
     * @brief Maps `TensorDataType::BF16` to CUDA `__nv_bfloat16`
     */
    template<> struct TensorDataTypeMap<TensorDataType::BF16> { using type = __nv_bfloat16; };

    /**
     * @brief Maps `TensorDataType::FP8_E4M3` to CUDA `__nv_fp8_e4m3`
     */
    template<> struct TensorDataTypeMap<TensorDataType::FP8_E4M3> { using type = __nv_fp8_e4m3; };

    /**
     * @brief Maps `TensorDataType::FP8_E5M2` to CUDA `__nv_fp8_e5m2`
     */
    template<> struct TensorDataTypeMap<TensorDataType::FP8_E5M2> { using type = __nv_fp8_e5m2; };

    /**
     * @brief Maps `TensorDataType::INT8` to `std::int8_t`
     */
    template<> struct TensorDataTypeMap<TensorDataType::INT8> { using type = std::int8_t; };

    /**
     * @brief Maps `TensorDataType::INT16` to `std::int16_t`
     */
    template<> struct TensorDataTypeMap<TensorDataType::INT16> { using type = std::int16_t; };

    /**
     * @brief Maps `TensorDataType::INT32` to `std::int32_t`
     */
    template<> struct TensorDataTypeMap<TensorDataType::INT32> { using type = std::int32_t; };

    /**
     * @brief Maps `TensorDataType::UINT8` to `std::uint8_t`
     */
    template<> struct TensorDataTypeMap<TensorDataType::UINT8> { using type = std::uint8_t; };

    /**
     * @brief Maps `TensorDataType::UINT16` to `std::uint16_t`
     */
    template<> struct TensorDataTypeMap<TensorDataType::UINT16> { using type = std::uint16_t; };

    /**
     * @brief Maps `TensorDataType::UINT32` to `std::uint32_t`
     */
    template<> struct TensorDataTypeMap<TensorDataType::UINT32> { using type = std::uint32_t; };

    /**
     * @brief Compile-time mapping from abstract TensorDataType -> host-compatible type
     *
     * Host mapping indicates the preferred C++ type to use on the host when reading,
     * writing, or converting tensor element values. For many reduced-precision or
     * device-only types the host mapping is `float` (e.g. FP16, BF16, FP8) to simplify
     * conversions and numerical handling on the CPU.
     *
     * @tparam TDataType Abstract tensor data type to map
     */
    export template<TensorDataType TDataType>
        struct HostDataTypeMap {
        static_assert(TDataType != TDataType, "Unsupported host data type mapping (HostDataTypeMap)");
    };

    /** Concrete host-compatible type specializations */

    /**
     * @brief Host type for `TensorDataType::FP32`
     */
    template<> struct HostDataTypeMap<TensorDataType::FP32> { using type = float; };

    /**
     * @brief Host type for `TensorDataType::FP16` (use `float` on host)
     */
    template<> struct HostDataTypeMap<TensorDataType::FP16> { using type = float; };

    /**
     * @brief Host type for `TensorDataType::BF16` (use `float` on host)
     */
    template<> struct HostDataTypeMap<TensorDataType::BF16> { using type = float; };

    /**
     * @brief Host type for `TensorDataType::FP8_E4M3` (use `float` on host)
     */
    template<> struct HostDataTypeMap<TensorDataType::FP8_E4M3> { using type = float; };

    /**
     * @brief Host type for `TensorDataType::FP8_E5M2` (use `float` on host)
     */
    template<> struct HostDataTypeMap<TensorDataType::FP8_E5M2> { using type = float; };

    /**
     * @brief Host type for `TensorDataType::INT8`
     */
    template<> struct HostDataTypeMap<TensorDataType::INT8> { using type = std::int8_t; };

    /**
     * @brief Host type for `TensorDataType::INT16`
     */
    template<> struct HostDataTypeMap<TensorDataType::INT16> { using type = std::int16_t; };

    /**
     * @brief Host type for `TensorDataType::INT32`
     */
    template<> struct HostDataTypeMap<TensorDataType::INT32> { using type = std::int32_t; };

    /**
     * @brief Host type for `TensorDataType::UINT8`
     */
    template<> struct HostDataTypeMap<TensorDataType::UINT8> { using type = std::uint8_t; };

    /**
     * @brief Host type for `TensorDataType::UINT16`
     */
    template<> struct HostDataTypeMap<TensorDataType::UINT16> { using type = std::uint16_t; };

    /**
     * @brief Host type for `TensorDataType::UINT32`
     */
    template<> struct HostDataTypeMap<TensorDataType::UINT32> { using type = std::uint32_t; };

}