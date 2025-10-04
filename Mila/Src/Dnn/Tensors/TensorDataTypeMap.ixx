/**
 * @file TensorDataTypeMap.ixx
 * @brief Concrete C++ type to abstract TensorDataType mapping utilities
 *
 * This module provides a minimal, explicit mapping from concrete C++ element
 * types to the abstract `TensorDataType` enumeration. It is used by migration
 * and helper utilities that must convert a concrete element type into the
 * corresponding abstract TensorDataType value (for example, when adapting
 * legacy APIs or performing compile-time dispatch).
 *
 * Key notes:
 * - The primary template emits a clear compile-time diagnostic for unsupported
 *   concrete types; only provided specializations are valid.
 * - This mapping is a small migration helper and not intended to replace the
 *   canonical `TensorDataTypeTraits` which defines per-enum compile-time traits.
 */

module;
#include <type_traits>
#include <cstdint>

export module Dnn.TensorTypeMap;

import Dnn.TensorDataType;

namespace Mila::Dnn
{
    // Helper to produce nice dependent static_assert messages
    template <typename T>
    struct dependent_false : std::false_type {};

    /**
     * @brief Primary template for mapping concrete C++ types to TensorDataType
     *
     * Provides a compile-time mapping from a concrete C++ element type to the
     * corresponding abstract `TensorDataType` enumeration value. Only specializations
     * for supported concrete types should be provided. Instantiating the primary
     * template for unsupported types produces a clear static_assert failure.
     *
     * @tparam TElementType The concrete C++ tensor element type
     */
    export template <typename TElementType>
    struct TensorDataTypeMap {
        static_assert(dependent_false<TElementType>::value,
                      "Unsupported concrete type for TensorDataTypeMap");
    };

    /**
     * @brief Concrete type mapping for float (FP32)
     */
    template <>
    struct TensorDataTypeMap<float> {
        static constexpr TensorDataType data_type = TensorDataType::FP32;
    };

    /**
     * @brief Concrete type mapping for 8-bit signed integer
     */
    template <>
    struct TensorDataTypeMap<std::int8_t> {
        static constexpr TensorDataType data_type = TensorDataType::INT8;
    };

    /**
     * @brief Concrete type mapping for 8-bit unsigned integer
     */
    template <>
    struct TensorDataTypeMap<std::uint8_t> {
        static constexpr TensorDataType data_type = TensorDataType::UINT8;
    };

    /**
     * @brief Concrete type mapping for 16-bit signed integer
     */
    template <>
    struct TensorDataTypeMap<std::int16_t> {
        static constexpr TensorDataType data_type = TensorDataType::INT16;
    };

    /**
     * @brief Concrete type mapping for 32-bit signed integer
     *
     * Use fixed-width `std::int32_t` to ensure deterministic mapping across platforms.
     */
    template <>
    struct TensorDataTypeMap<std::int32_t> {
        static constexpr TensorDataType data_type = TensorDataType::INT32;
    };

    /**
     * @brief Concrete type mapping for 16-bit unsigned integer
     */
    template <>
    struct TensorDataTypeMap<std::uint16_t> {
        static constexpr TensorDataType data_type = TensorDataType::UINT16;
    };

    /**
     * @brief Concrete type mapping for 32-bit unsigned integer
     */
    template <>
    struct TensorDataTypeMap<std::uint32_t> {
        static constexpr TensorDataType data_type = TensorDataType::UINT32;
    };
}