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
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

    // TODO: Add remaining integer type specializations (INT4, INT8, UINT4, UINT8, UINT16, UINT32)

    /**
     * @brief Concept constraining TensorDataType to floating-point types
     *
     * Validates at compile-time that the specified TensorDataType represents
     * a floating-point format. Used to enforce type safety in floating-point
     * specific operations and algorithms.
     *
     * @tparam TDataType The TensorDataType to validate
     */
    export template<TensorDataType TDataType>
        concept FloatTensorDataType = TensorDataTypeTraits<TDataType>::is_float_type;

    /**
     * @brief Concept constraining TensorDataType to integer types
     *
     * Validates at compile-time that the specified TensorDataType represents
     * an integer format. Used to enforce type safety in integer-specific
     * operations and prevent inappropriate mixed-type operations.
     *
     * @tparam TDataType The TensorDataType to validate
     */
    export template<TensorDataType TDataType>
        concept IntegerTensorDataType = TensorDataTypeTraits<TDataType>::is_integer_type;

    /**
     * @brief Concept constraining TensorDataType to host-compatible types
     *
     * Validates that the specified TensorDataType can be used in host (CPU)
     * environments without requiring device-specific compilation or runtime.
     * Essential for operations that must execute on the host.
     *
     * @tparam TDataType The TensorDataType to validate
     */
    export template<TensorDataType TDataType>
        concept HostCompatibleDataType = !TensorDataTypeTraits<TDataType>::is_device_only;

    /**
     * @brief Concept constraining TensorDataType to device-only types
     *
     * Validates that the specified TensorDataType requires device-accessible
     * memory and cannot be directly used in host-only contexts. Used to
     * enforce proper memory resource selection for device-specific types.
     *
     * @tparam TDataType The TensorDataType to validate
     */
    export template<TensorDataType TDataType>
        concept DeviceOnlyDataType = TensorDataTypeTraits<TDataType>::is_device_only;

    /**
 * @brief Calculates storage requirements for tensor elements
 *
 * Computes the actual memory storage required for a given number of tensor
 * elements. Essential for accurate memory allocation and buffer sizing.
 *
 * @tparam TDataType The tensor data type specifying storage format
 * @param element_count Number of logical tensor elements to store
 * @return Number of bytes required for storage
 *
 * @note For all current data types, returns element_count * size_in_bytes
 * @note Future packed types will require special handling
 */
    export template<TensorDataType TDataType>
        constexpr size_t getStorageSize( size_t element_count ) {
        return element_count * TensorDataTypeTraits<TDataType>::size_in_bytes;
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
    template <typename T>
    struct dependent_false : std::false_type {};

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
    export template <typename T>
        constexpr TensorDataType getTensorDataTypeEnum() {
        if constexpr ( std::is_same_v<T, float> ) return TensorDataType::FP32;
        else if constexpr ( std::is_same_v<T, half> ) return TensorDataType::FP16;
        else if constexpr ( std::is_same_v<T, nv_bfloat16> ) return TensorDataType::BF16;
        else if constexpr ( std::is_same_v<T, __nv_fp8_e4m3> ) return TensorDataType::FP8_E4M3;
        else if constexpr ( std::is_same_v<T, __nv_fp8_e5m2> ) return TensorDataType::FP8_E5M2;
        else if constexpr ( std::is_same_v<T, int8_t> ) return TensorDataType::INT8;
        else if constexpr ( std::is_same_v<T, int16_t> ) return TensorDataType::INT16;
        else if constexpr ( std::is_same_v<T, int32_t> ) return TensorDataType::INT32;
        else if constexpr ( std::is_same_v<T, uint8_t> ) return TensorDataType::UINT8;
        else if constexpr ( std::is_same_v<T, uint16_t> ) return TensorDataType::UINT16;
        else if constexpr ( std::is_same_v<T, uint32_t> ) return TensorDataType::UINT32;
        else {
            static_assert(dependent_false<T>::value, "Unsupported tensor data type");
            return TensorDataType::FP32;
        }
    }
}