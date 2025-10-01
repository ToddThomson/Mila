/**
 * @file TensorDataType.ixx
 * @brief Abstract tensor data type enumeration and traits system for device-agnostic tensor operations
 *
 * This module provides a comprehensive type abstraction layer that enables tensor operations
 * across different compute devices (CPU, CUDA, Metal, OpenCL) without exposing device-specific
 * concrete types to host compilation. The system supports standard floating-point and integer
 * types as well as advanced precision formats including FP8, FP4, and packed sub-byte types.
 */

module;
#include <string>
#include <cstdint>
#include <type_traits>

export module Dnn.TensorDataType;

namespace Mila::Dnn
{
    /**
     * @brief Enumeration of supported abstract tensor data types
     *
     * Defines device-agnostic tensor data types that can be mapped to concrete
     * implementations on different compute devices. This abstraction prevents
     * host compilation issues with device-specific types while enabling
     * compile-time dispatch and optimization.
     *
     * Supported categories:
     * - Standard floating-point: FP32
     * - Reduced precision floating-point: FP16, BF16, FP8_E4M3, FP8_E5M2
     * - Integer types: Various widths from 8-bit to 32-bit, signed and unsigned
     *
     * @note Device-only types (FP16, BF16, FP8) require device-accessible memory
     * @note Packed sub-byte types (FP4, INT4, UINT4) are planned for future implementation
     */
    export enum class TensorDataType {
        FP32,     ///< 32-bit IEEE 754 floating point, host-compatible
        FP16,     ///< 16-bit half precision floating point, device-only
        BF16,     ///< 16-bit brain floating point, device-only
        FP8_E4M3, ///< 8-bit floating point with 4-bit exponent and 3-bit mantissa, device-only
        FP8_E5M2, ///< 8-bit floating point with 5-bit exponent and 2-bit mantissa, device-only
        
        // Future packed types (not yet implemented)
        // FP4_E2M1, ///< 4-bit floating point with 2-bit exponent and 1-bit mantissa, packed - FUTURE
        // FP4_E3M0, ///< 4-bit floating point with 3-bit exponent and 0-bit mantissa, packed - FUTURE
        // INT4,     ///< 4-bit signed integer, packed - FUTURE
        // UINT4,    ///< 4-bit unsigned integer, packed - FUTURE
        
        INT8,     ///< 8-bit signed integer
        INT16,    ///< 16-bit signed integer, host-compatible
        INT32,    ///< 32-bit signed integer, host-compatible
        UINT8,    ///< 8-bit unsigned integer
        UINT16,   ///< 16-bit unsigned integer, host-compatible
        UINT32,   ///< 32-bit unsigned integer, host-compatible
    };

    /**
     * @brief Converts TensorDataType enumeration to human-readable string
     */
    export inline std::string tensorDataTypeToString( TensorDataType type ) {
        switch ( type ) {
            case TensorDataType::FP32: return "FP32";
            case TensorDataType::FP16: return "FP16";
            case TensorDataType::BF16: return "BF16";
            case TensorDataType::FP8_E4M3: return "FP8_E4M3";
            case TensorDataType::FP8_E5M2: return "FP8_E5M2";
            // Future packed types commented out
            // case TensorDataType::FP4_E2M1: return "FP4_E2M1";
            // case TensorDataType::FP4_E3M0: return "FP4_E3M0";
            // case TensorDataType::INT4: return "INT4";
            // case TensorDataType::UINT4: return "UINT4";
            case TensorDataType::INT8: return "INT8";
            case TensorDataType::INT16: return "INT16";
            case TensorDataType::INT32: return "INT32";
            case TensorDataType::UINT8: return "UINT8";
            case TensorDataType::UINT16: return "UINT16";
            case TensorDataType::UINT32: return "UINT32";
            default: return "Unknown";
        }
    }
    
    /**
     * @brief Helper template for compile-time assertion failures
     *
     * Enables static_assert to fail only when template is instantiated,
     * allowing conditional compilation based on template parameters.
     * Used in getTensorDataTypeEnum for unsupported type detection.
     *
     * @tparam T Type parameter for instantiation-dependent failure
     */
    /*template <typename T>
    struct dependent_false : std::false_type {};*/

    /**
     * @brief Maps concrete C++ types to abstract TensorDataType enumeration
     *
     * Provides reverse mapping from device-specific concrete types to abstract
     * TensorDataType enumeration values. Maintains compatibility with existing
     * TensorTrait system during migration to abstract type system.
     *
     * @tparam T Concrete C++ type to map (float, half, int32_t, etc.)
     * @return Corresponding TensorDataType enumeration value
     *
     * @throws Compilation error via static_assert for unsupported types
     *
     * @note This function maintains backward compatibility during migration
     * @note Device-specific types (half, nv_bfloat16) map to device-only enums
     * @note Host types (float, int32_t) map to host-compatible enums
     *
     * Supported mappings:
     * - float ? TensorDataType::FP32
     * - half ? TensorDataType::FP16
     * - nv_bfloat16 ? TensorDataType::BF16
     * - __nv_fp8_e4m3 ? TensorDataType::FP8_E4M3
     * - __nv_fp8_e5m2 ? TensorDataType::FP8_E5M2
     * - int16_t ? TensorDataType::INT16
     * - int32_t ? TensorDataType::INT32
     * - uint16_t ? TensorDataType::UINT16
     * - uint32_t ? TensorDataType::UINT32
     */
    //export template <typename T>
    //    constexpr TensorDataType getTensorDataTypeEnum() {
    //    if constexpr ( std::is_same_v<T, float> ) return TensorDataType::FP32;
    //    else if constexpr ( std::is_same_v<T, half> ) return TensorDataType::FP16;
    //    else if constexpr ( std::is_same_v<T, nv_bfloat16> ) return TensorDataType::BF16;
    //    else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) return TensorDataType::FP8_E4M3;
    //    else if constexpr ( std::is_same_v<T, __nv_fp8_e5m2> ) return TensorDataType::FP8_E5M2;
    //    else if constexpr ( std::is_same_v<T, int8_t> ) return TensorDataType::INT8;
    //    else if constexpr ( std::is_same_v<T, int16_t> ) return TensorDataType::INT16;
    //    else if constexpr ( std::is_same_v<T, int32_t> ) return TensorDataType::INT32;
    //    else if constexpr ( std::is_same_v<T, uint8_t> ) return TensorDataType::UINT8;
    //    else if constexpr ( std::is_same_v<T, uint16_t> ) return TensorDataType::UINT16;
    //    else if constexpr ( std::is_same_v<T, uint32_t> ) return TensorDataType::UINT32;
    //    else {
    //        static_assert(dependent_false<T>::value, "Unsupported tensor data type");
    //        return TensorDataType::FP32;
    //    }
    //}
}