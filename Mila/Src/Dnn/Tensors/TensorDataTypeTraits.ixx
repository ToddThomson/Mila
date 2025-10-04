/**
 * @file TensorDataTypeTraits.ixx
 * @brief Compile-time traits for the abstract TensorDataType enumeration.
 *
 * Provides per-type properties used throughout the tensor system:
 * - size and alignment
 * - host/device classification flags
 * - human-readable type name
 * - host conversion type used when converting values to/from host representations
 */

module;
#include <cstdint>
#include <type_traits>
#include <string_view>

export module Dnn.TensorTypeTraits;

import Dnn.TensorDataType;
import Compute.MemoryResource;

namespace Mila::Dnn
{
    /**
     * @brief Compile-time traits for TensorDataType enumeration values
     *
     * Primary template providing compile-time characteristics for each abstract
     * tensor data type. Specializations define properties including storage size,
     * alignment requirements, device compatibility, and type classification.
     *
     * This traits system enables:
     * - Compile-time memory layout calculations
     * - Device compatibility validation
     * - Type-safe tensor operations
     * - Optimal memory allocation strategies
     *
     * @tparam TDataType The TensorDataType enumeration value to query
     *
     * @note This is a primary template that must be specialized for each supported type
     * @note Accessing an unspecialized version will result in compilation error
     *
     * @see TensorDataType for supported enumeration values
     */
    export template<TensorDataType TDataType>
        struct TensorDataTypeTraits;

    /**
     * @brief Traits specialization for 32-bit IEEE 754 floating point
     *
     * Standard single-precision floating point compatible with both host and device
     * environments. Provides maximum compatibility across all compute platforms.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::FP32> {
        static constexpr bool is_float_type = true;     ///< Floating-point type classification
        static constexpr bool is_integer_type = false;  ///< Not an integer type
        static constexpr bool is_device_only = false;   ///< Compatible with host environments
        static constexpr size_t size_in_bytes = 4;      ///< Memory footprint per element
        static constexpr size_t alignment = 4;          ///< Required memory alignment
        static constexpr const char* type_name = "FP32"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 16-bit half precision floating point
     *
     * IEEE 754-2008 binary16 format providing reduced precision with smaller memory
     * footprint. Requires device-accessible memory due to limited host support.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::FP16> {
        static constexpr bool is_float_type = true;     ///< Floating-point type classification
        static constexpr bool is_integer_type = false;  ///< Not an integer type
        static constexpr bool is_device_only = true;    ///< Requires device-accessible memory
        static constexpr size_t size_in_bytes = 2;      ///< Memory footprint per element
        static constexpr size_t alignment = 2;          ///< Required memory alignment
        static constexpr const char* type_name = "FP16"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 16-bit brain floating point
     *
     * Google Brain's bfloat16 format with same dynamic range as FP32 but reduced
     * precision. Optimized for machine learning workloads with better numerical
     * stability than FP16 for gradient computations.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::BF16> {
        static constexpr bool is_float_type = true;     ///< Floating-point type classification
        static constexpr bool is_integer_type = false;  ///< Not an integer type
        static constexpr bool is_device_only = true;    ///< Requires device-accessible memory
        static constexpr size_t size_in_bytes = 2;      ///< Memory footprint per element
        static constexpr size_t alignment = 2;          ///< Required memory alignment
        static constexpr const char* type_name = "BF16"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 8-bit floating point with E4M3 format
     *
     * Ultra-low precision format with 4-bit exponent and 3-bit mantissa.
     * Optimized for inference workloads where extreme memory efficiency
     * is prioritized over numerical precision.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::FP8_E4M3> {
        static constexpr bool is_float_type = true;      ///< Floating-point type classification
        static constexpr bool is_integer_type = false;   ///< Not an integer type
        static constexpr bool is_device_only = true;     ///< Requires device-accessible memory
        static constexpr size_t size_in_bytes = 1;       ///< Memory footprint per element
        static constexpr size_t alignment = 1;           ///< Required memory alignment
        static constexpr const char* type_name = "FP8_E4M3"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 8-bit floating point with E5M2 format
     *
     * Ultra-low precision format with 5-bit exponent and 2-bit mantissa.
     * Provides wider dynamic range than E4M3 at the cost of mantissa precision.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::FP8_E5M2> {
        static constexpr bool is_float_type = true;      ///< Floating-point type classification
        static constexpr bool is_integer_type = false;   ///< Not an integer type
        static constexpr bool is_device_only = true;     ///< Requires device-accessible memory
        static constexpr size_t size_in_bytes = 1;       ///< Memory footprint per element
        static constexpr size_t alignment = 1;           ///< Required memory alignment
        static constexpr const char* type_name = "FP8_E5M2"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 16-bit signed integer
     *
     * Standard signed integer type compatible with both host and device environments.
     * Provides good balance between range and memory efficiency for integer tensors.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::INT16> {
        static constexpr bool is_float_type = false;     ///< Not a floating-point type
        static constexpr bool is_integer_type = true;    ///< Integer type classification
        static constexpr bool is_device_only = false;    ///< Compatible with host environments
        static constexpr size_t size_in_bytes = 2;       ///< Memory footprint per element
        static constexpr size_t alignment = 2;           ///< Required memory alignment
        static constexpr const char* type_name = "INT16"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 32-bit signed integer
     *
     * Standard signed integer type with maximum compatibility across platforms.
     * Provides full integer range for applications requiring precise integer arithmetic.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::INT32> {
        static constexpr bool is_float_type = false;     ///< Not a floating-point type
        static constexpr bool is_integer_type = true;    ///< Integer type classification
        static constexpr bool is_device_only = false;    ///< Compatible with host environments
        static constexpr size_t size_in_bytes = 4;       ///< Memory footprint per element
        static constexpr size_t alignment = 4;           ///< Required memory alignment
        static constexpr const char* type_name = "INT32"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 8-bit signed integer
     *
     * Standard signed byte type compatible with both host and device environments.
     * Commonly used for quantized models and compact integer representations.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::INT8> {
        static constexpr bool is_float_type = false;     ///< Not a floating-point type
        static constexpr bool is_integer_type = true;    ///< Integer type classification
        static constexpr bool is_device_only = false;    ///< Compatible with host environments
        static constexpr size_t size_in_bytes = 1;       ///< Memory footprint per element
        static constexpr size_t alignment = 1;           ///< Required memory alignment
        static constexpr const char* type_name = "INT8"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 8-bit unsigned integer
     *
     * Standard unsigned byte type compatible with both host and device environments.
     * Ideal for image data, masks, and compact positive integer representations.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::UINT8> {
        static constexpr bool is_float_type = false;     ///< Not a floating-point type
        static constexpr bool is_integer_type = true;    ///< Integer type classification
        static constexpr bool is_device_only = false;    ///< Compatible with host environments
        static constexpr size_t size_in_bytes = 1;       ///< Memory footprint per element
        static constexpr size_t alignment = 1;           ///< Required memory alignment
        static constexpr const char* type_name = "UINT8"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 16-bit unsigned integer
     *
     * Standard unsigned short type compatible with both host and device environments.
     * Provides good balance between range and memory efficiency for positive integer tensors.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::UINT16> {
        static constexpr bool is_float_type = false;     ///< Not a floating-point type
        static constexpr bool is_integer_type = true;    ///< Integer type classification
        static constexpr bool is_device_only = false;    ///< Compatible with host environments
        static constexpr size_t size_in_bytes = 2;       ///< Memory footprint per element
        static constexpr size_t alignment = 2;           ///< Required memory alignment
        static constexpr const char* type_name = "UINT16"; ///< Human-readable type identifier
    };

    /**
     * @brief Traits specialization for 32-bit unsigned integer
     *
     * Standard unsigned integer type with maximum compatibility across platforms.
     * Provides full positive integer range for applications requiring large unsigned values.
     */
    template<>
    struct TensorDataTypeTraits<TensorDataType::UINT32> {
        static constexpr bool is_float_type = false;     ///< Not a floating-point type
        static constexpr bool is_integer_type = true;    ///< Integer type classification
        static constexpr bool is_device_only = false;    ///< Compatible with host environments
        static constexpr size_t size_in_bytes = 4;       ///< Memory footprint per element
        static constexpr size_t alignment = 4;           ///< Required memory alignment
        static constexpr const char* type_name = "UINT32"; ///< Human-readable type identifier
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