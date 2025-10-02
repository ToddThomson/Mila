/**
 * @file TensorHostTypeMap.ixx
 * @brief Device-agnostic host type mapping for abstract TensorDataType enumeration
 *
 * This module provides centralized mapping from abstract tensor data types to
 * appropriate host-compatible C++ types for conversion and host-side processing.
 * These mappings are device-agnostic and define the standard host representation
 * for each abstract tensor data type across all compute backends.
 *
 * Key architectural principles:
 * - Device-agnostic: Host types are consistent across all compute devices
 * - Conversion-oriented: Optimized for host-device data transfers
 * - Direct type mappings: Preserves precision and signedness where possible
 * - Single source of truth: Eliminates duplication across device-specific files
 *
 * Usage patterns:
 * - Host-device data transfers and conversions
 * - Fill operations using host-provided values
 * - Debugging and inspection of tensor values
 * - External API integration requiring standard C++ types
 *
 * Design rationale:
 * - Direct mappings preserve type characteristics (precision, signedness)
 * - Floating-point types use appropriate host representations
 * - Integer types maintain their original bit width and signedness
 * - Maps are closed via explicit specializations with clear error messages
 */

module;
#include <cstdint>
#include <type_traits>

export module Dnn.TensorHostTypeMap;

import Dnn.TensorDataType;

namespace Mila::Dnn
{
    /**
     * @brief Maps abstract TensorDataType to host-compatible C++ type and TensorDataType
     *
     * Defines the preferred host-side type and corresponding TensorDataType for each
     * abstract tensor data type, used for host-device conversions, fill operations,
     * and host-side processing. This mapping preserves type characteristics where
     * possible while ensuring host compatibility.
     *
     * Design decisions:
     * - Floating-point device types (FP16, BF16, FP8) map to `float` for host compatibility
     * - Integer types map to their corresponding fixed-width standard types
     * - Maintains precision and signedness characteristics of original types
     * - Provides direct host representation for device-agnostic operations
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     *
     * @note This is a closed template - only explicit specializations are valid
     * @note Host types are device-agnostic and consistent across all compute backends
     * @note Instantiating the primary template produces a clear compile-time error
     */
    export template<TensorDataType TDataType>
        struct TensorHostTypeMap {
        static_assert(TDataType != TDataType, "Unsupported TensorDataType for host type mapping");
    };

    // ====================================================================
    // Floating-Point Type Specializations
    // ====================================================================

    /**
     * @brief Host type for 32-bit IEEE 754 floating point
     *
     * Direct mapping as FP32 is natively host-compatible.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::FP32> {
        using host_type = float;
        static constexpr TensorDataType host_data_type = TensorDataType::FP32;
    };

    /**
     * @brief Host type for 16-bit half precision floating point
     *
     * Maps to float for host compatibility - device-specific precision
     * is preserved in device memory, but host operations use float.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::FP16> {
        using host_type = float;
        static constexpr TensorDataType host_data_type = TensorDataType::FP32;
    };

    /**
     * @brief Host type for 16-bit brain floating point
     *
     * Maps to float for host compatibility - bfloat16 is primarily
     * a device optimization for training workloads.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::BF16> {
        using host_type = float;
        static constexpr TensorDataType host_data_type = TensorDataType::FP32;
    };

    /**
     * @brief Host type for 8-bit floating point with E4M3 format
     *
     * Maps to float for host compatibility - FP8 is an ultra-low
     * precision format for specialized inference workloads.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::FP8_E4M3> {
        using host_type = float;
        static constexpr TensorDataType host_data_type = TensorDataType::FP32;
    };

    /**
     * @brief Host type for 8-bit floating point with E5M2 format
     *
     * Maps to float for host compatibility - FP8 is an ultra-low
     * precision format for specialized inference workloads.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::FP8_E5M2> {
        using host_type = float;
        static constexpr TensorDataType host_data_type = TensorDataType::FP32;
    };

    // ====================================================================
    // Integer Type Specializations
    // ====================================================================

    /**
     * @brief Host type for 8-bit signed integer
     *
     * Direct mapping preserving the 8-bit signed integer characteristics
     * for accurate host-side representation and operations.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::INT8> {
        using host_type = std::int8_t;
        static constexpr TensorDataType host_data_type = TensorDataType::INT8;
    };

    /**
     * @brief Host type for 16-bit signed integer
     *
     * Direct mapping preserving the 16-bit signed integer characteristics
     * for accurate host-side representation and operations.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::INT16> {
        using host_type = std::int16_t;
        static constexpr TensorDataType host_data_type = TensorDataType::INT16;
    };

    /**
     * @brief Host type for 32-bit signed integer
     *
     * Direct mapping as INT32 is natively host-compatible.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::INT32> {
        using host_type = std::int32_t;
        static constexpr TensorDataType host_data_type = TensorDataType::INT32;
    };

    /**
     * @brief Host type for 8-bit unsigned integer
     *
     * Direct mapping preserving the 8-bit unsigned integer characteristics
     * for accurate host-side representation and operations.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::UINT8> {
        using host_type = std::uint8_t;
        static constexpr TensorDataType host_data_type = TensorDataType::UINT8;
    };

    /**
     * @brief Host type for 16-bit unsigned integer
     *
     * Direct mapping preserving the 16-bit unsigned integer characteristics
     * for accurate host-side representation and operations.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::UINT16> {
        using host_type = std::uint16_t;
        static constexpr TensorDataType host_data_type = TensorDataType::UINT16;
    };

    /**
     * @brief Host type for 32-bit unsigned integer
     *
     * Direct mapping preserving the 32-bit unsigned integer characteristics
     * for accurate host-side representation and operations.
     */
    template<>
    struct TensorHostTypeMap<TensorDataType::UINT32> {
        using host_type = std::uint32_t;
        static constexpr TensorDataType host_data_type = TensorDataType::UINT32;
    };

    // ====================================================================
    // Convenience Aliases and Utilities
    // ====================================================================

    /**
     * @brief Convenience alias for accessing host type mapping
     *
     * Provides a more concise way to access the host type for a given
     * abstract tensor data type, following modern C++ alias template patterns.
     *
     * @tparam TDataType Abstract tensor data type
     *
     * Example usage:
     * ```cpp
     * using HostType = host_type_t<TensorDataType::FP16>;  // -> float
     * using IntHostType = host_type_t<TensorDataType::INT8>;  // -> std::int8_t
     * ```
     */
    export template<TensorDataType TDataType>
        using host_type_t = typename TensorHostTypeMap<TDataType>::host_type;

    /**
     * @brief Convenience alias for accessing host TensorDataType mapping
     *
     * Provides a more concise way to access the host TensorDataType for a given
     * abstract tensor data type, simplifying toHost() implementations.
     *
     * @tparam TDataType Abstract tensor data type
     *
     * Example usage:
     * ```cpp
     * constexpr auto HostDataType = host_data_type_v<TensorDataType::FP16>;  // -> TensorDataType::FP32
     * constexpr auto IntHostDataType = host_data_type_v<TensorDataType::INT8>;  // -> TensorDataType::INT8
     * ```
     */
    export template<TensorDataType TDataType>
        constexpr TensorDataType host_data_type_v = TensorHostTypeMap<TDataType>::host_data_type;

    /**
     * @brief Checks if a TensorDataType maps to a floating-point host type
     *
     * Compile-time utility to determine if the host representation of an
     * abstract tensor data type is a floating-point type.
     *
     * @tparam TDataType Abstract tensor data type to check
     */
    export template<TensorDataType TDataType>
        constexpr bool is_host_float_type = std::is_floating_point_v<host_type_t<TDataType>>;

    /**
     * @brief Checks if a TensorDataType maps to an integer host type
     *
     * Compile-time utility to determine if the host representation of an
     * abstract tensor data type is an integer type.
     *
     * @tparam TDataType Abstract tensor data type to check
     */
    export template<TensorDataType TDataType>
        constexpr bool is_host_integer_type = std::is_integral_v<host_type_t<TDataType>>;
}