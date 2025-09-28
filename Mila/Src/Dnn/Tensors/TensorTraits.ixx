/**
 * @file TensorTraits.ixx
 * @brief Tensor validation and migration utilities for abstract data type system
 *
 * This module provides validation concepts for the abstract TensorDataType system
 * and minimal concrete type traits to support migration utilities. The primary
 * purpose is to bridge the abstract data type enumeration with memory resource
 * validation for compile-time tensor configuration checking.
 *
 * Key architectural principles:
 * - Primary validation works with abstract TensorDataType enumeration
 * - Concrete type traits exist only to support migration utilities
 * - Clean separation between validation logic and data type definitions
 * - No legacy validation concepts - forward-looking design only
 */

module;
#include <type_traits>
#include <cstdint>
#include <string_view>

export module Dnn.TensorTraits;

import Dnn.TensorDataType;
import Compute.MemoryResource;

namespace Mila::Dnn
{
    /**
     * @brief Primary template for concrete type to abstract type mapping
     *
     * This trait structure provides mapping from concrete C++ types to abstract
     * TensorDataType enumeration values. Used exclusively by migration utility
     * functions like getTensorDataTypeEnum<T>().
     *
     * @tparam TElementType The concrete C++ tensor element type
     *
     * @note This is intentionally minimal - only supports getTensorDataTypeEnum utility
     * @note Primary template is undefined - only specializations are valid
     */
    export template <typename TElementType>
        struct TensorTrait;

    /**
     * @brief Concrete type mapping for float (FP32)
     */
    template <>
    struct TensorTrait<float> {
        static constexpr TensorDataType data_type = TensorDataType::FP32;
    };
    
    /**
     * @brief Concrete type mapping for 16-bit signed integer
     */
    template <>
    struct TensorTrait<int16_t> {
        static constexpr TensorDataType data_type = TensorDataType::INT16;
    };

    /**
     * @brief Concrete type mapping for 32-bit signed integer
     */
    template <>
    struct TensorTrait<int> {
        static constexpr TensorDataType data_type = TensorDataType::INT32;
    };

    /**
     * @brief Concrete type mapping for 16-bit unsigned integer
     */
    template <>
    struct TensorTrait<uint16_t> {
        static constexpr TensorDataType data_type = TensorDataType::UINT16;
    };

    /**
     * @brief Concrete type mapping for 32-bit unsigned integer
     */
    template <>
    struct TensorTrait<uint32_t> {
        static constexpr TensorDataType data_type = TensorDataType::UINT32;
    };

    // ====================================================================
    // Primary Tensor Validation Concept for Abstract Data Types
    // ====================================================================

    /**
     * @brief Primary tensor configuration validation concept
     *
     * Validates that a TensorDataType and MemoryResource combination represents
     * a valid tensor configuration. This is the core validation concept for the
     * abstract tensor data type system.
     *
     * Validation ensures:
     * - Memory resource inherits from base MemoryResource class
     * - Device-only data types use device-accessible memory resources
     * - TensorDataTypeTraits is properly specialized for the data type
     * - All required trait properties are available at compile time
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration
     * @tparam TMemoryResource Memory resource type for tensor storage
     *
     * @note This is the primary validation interface for the tensor system
     * @note Replaces all legacy concrete type validation concepts
     * @note Used directly by Tensor class template constraints
     *
     * @see TensorDataType for supported data type enumeration
     * @see TensorDataTypeTraits for compile-time data type characteristics
     * @see MemoryResource for memory resource base class
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        concept isValidTensor =
        // Memory resource must inherit from base class
        std::is_base_of_v<Compute::MemoryResource, TMemoryResource> &&

        // Device-only types must use device-accessible memory
        (!TensorDataTypeTraits<TDataType>::is_device_only || TMemoryResource::is_device_accessible) &&

        // Ensure TensorDataTypeTraits is properly specialized with required members
        requires {
            // Type must be specialized
            typename TensorDataTypeTraits<TDataType>;

            // Required compile-time properties
            TensorDataTypeTraits<TDataType>::size_in_bytes;
            TensorDataTypeTraits<TDataType>::is_device_only;
            TensorDataTypeTraits<TDataType>::is_float_type;
            TensorDataTypeTraits<TDataType>::is_integer_type;
            TensorDataTypeTraits<TDataType>::type_name;
            TensorDataTypeTraits<TDataType>::alignment;
        };

    /**
     * @brief Concept constraining abstract data types to floating-point formats
     *
     * Validates at compile-time that the specified TensorDataType represents
     * a floating-point format for use in floating-point specific operations.
     *
     * @tparam TDataType The TensorDataType to validate for floating-point classification
     */
    export template<TensorDataType TDataType>
        concept ValidFloatTensorDataType = TensorDataTypeTraits<TDataType>::is_float_type;

    /**
     * @brief Concept constraining abstract data types to integer formats
     *
     * Validates at compile-time that the specified TensorDataType represents
     * an integer format for use in integer-specific operations.
     *
     * @tparam TDataType The TensorDataType to validate for integer classification
     */
    export template<TensorDataType TDataType>
        concept ValidIntegerTensorDataType = TensorDataTypeTraits<TDataType>::is_integer_type;

    /**
     * @brief Concept identifying device-only abstract data types
     *
     * Validates that the specified TensorDataType requires device-accessible
     * memory and cannot be used in host-only contexts.
     *
     * @tparam TDataType The TensorDataType to validate for device-only requirement
     */
    export template<TensorDataType TDataType>
        concept DeviceOnlyTensorDataType = TensorDataTypeTraits<TDataType>::is_device_only;

    /**
     * @brief Concept identifying host-compatible abstract data types
     *
     * Validates that the specified TensorDataType can be used in host (CPU)
     * environments without requiring device-specific compilation.
     *
     * @tparam TDataType The TensorDataType to validate for host compatibility
     */
    export template<TensorDataType TDataType>
        concept HostCompatibleTensorDataType = !TensorDataTypeTraits<TDataType>::is_device_only;
}