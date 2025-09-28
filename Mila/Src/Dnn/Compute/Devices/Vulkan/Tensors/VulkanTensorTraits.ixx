/**
 * @file VulkanTensorTraits.ixx
 * @brief Vulkan-specific tensor trait specializations for explicit graphics compute
 *
 * This module provides Vulkan-specific tensor characteristics and native type mappings
 * for explicit graphics and compute operations. Supports Vulkan's memory model with
 * fine-grained control over device memory allocation, synchronization, and compute
 * pipeline integration across different GPU vendors.
 */

module;
#include <cstdint>
#include <string_view>
#include <type_traits>

#ifdef VULKAN_AVAILABLE
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#endif

export module Compute.VulkanTensorTraits;

import Dnn.TensorTraits;
import Dnn.TensorDataType;
import Compute.DeviceType;

namespace Mila::Dnn
{
    // Vulkan uses standard C++ types but with explicit memory management
    // and compute pipeline integration requirements

    /**
     * @brief Vulkan-specific traits for abstract tensor data types
     *
     * Provides device-specific characteristics and native type mappings for
     * Vulkan-supported tensor data types, enabling type-safe value initialization
     * and explicit graphics/compute operations with fine-grained synchronization
     * control across different GPU architectures.
     *
     * Vulkan's explicit design requires careful consideration of:
     * - Memory heap selection and allocation strategy
     * - Command buffer recording and submission
     * - Pipeline barriers and synchronization primitives
     * - Device-local vs host-visible memory trade-offs
     */
    export class VulkanTensorTraits {
    public:
        /**
         * @brief Validates Vulkan support for abstract tensor data types
         *
         * Determines whether Vulkan devices and memory resources support the specified
         * abstract tensor data type, considering Vulkan's explicit memory model and
         * compute pipeline requirements across different GPU vendors and architectures.
         *
         * @tparam TDataType Abstract tensor data type to validate
         * @return true if Vulkan supports the data type, false otherwise
         */
        template<TensorDataType TDataType>
        static consteval bool supports() {
            return TDataType == TensorDataType::FP32 ||
                TDataType == TensorDataType::FP16 ||
                TDataType == TensorDataType::FP64 ||  // Requires VK_KHR_shader_float64
                TDataType == TensorDataType::INT8 ||
                TDataType == TensorDataType::INT16 ||
                TDataType == TensorDataType::INT32 ||
                TDataType == TensorDataType::INT64 ||
                TDataType == TensorDataType::UINT8 ||
                TDataType == TensorDataType::UINT16 ||
                TDataType == TensorDataType::UINT32 ||
                TDataType == TensorDataType::UINT64;
        }

    private:
        /**
         * @brief Helper function for compile-time native type deduction
         *
         * Maps abstract tensor data types to Vulkan-compatible native types,
         * using standard C++ types that align with SPIR-V and Vulkan shader
         * data type requirements. Handles both available and unavailable
         * Vulkan configurations with appropriate fallback types.
         */
        template<TensorDataType TDataType>
        static consteval auto get_native_type() {
            // Vulkan uses standard C++ types that map to SPIR-V types
            if constexpr ( TDataType == TensorDataType::FP32 ) return float{};
            else if constexpr ( TDataType == TensorDataType::FP16 ) {
                // Half precision requires VK_KHR_shader_float16_int8 extension
                return uint16_t{}; // Stored as 16-bit but requires special handling
            }
            else if constexpr ( TDataType == TensorDataType::FP64 ) {
                // Double precision requires VK_KHR_shader_float64 extension
                return double{};
            }
            else if constexpr ( TDataType == TensorDataType::INT8 ) return int8_t{};
            else if constexpr ( TDataType == TensorDataType::INT16 ) return int16_t{};
            else if constexpr ( TDataType == TensorDataType::INT32 ) return int32_t{};
            else if constexpr ( TDataType == TensorDataType::INT64 ) return int64_t{};
            else if constexpr ( TDataType == TensorDataType::UINT8 ) return uint8_t{};
            else if constexpr ( TDataType == TensorDataType::UINT16 ) return uint16_t{};
            else if constexpr ( TDataType == TensorDataType::UINT32 ) return uint32_t{};
            else if constexpr ( TDataType == TensorDataType::UINT64 ) return uint64_t{};
            else {
                static_assert(TDataType == TensorDataType::FP32,
                    "Unsupported tensor data type for Vulkan. Check VulkanTensorTraits::supports() for supported types.");
                return float{};
            }
        }

    public:
        /**
         * @brief Type alias for the native Vulkan type corresponding to a TensorDataType
         *
         * Provides compile-time mapping from abstract TensorDataType enumeration
         * to concrete C++ types compatible with Vulkan's SPIR-V shader requirements
         * and memory layout specifications.
         */
        template<TensorDataType TDataType>
        using native_type = decltype(get_native_type<TDataType>());

        /**
         * @brief Creates a native type value from a compatible input value
         *
         * Provides safe conversion from input values to the appropriate native
         * Vulkan-compatible type, handling cross-platform type conversions and
         * ensuring compatibility with SPIR-V data type requirements.
         *
         * @tparam TDataType Target abstract tensor data type
         * @tparam T Input value type
         * @param value Input value to convert
         * @return Native Vulkan-compatible type value suitable for compute operations
         */
        template<TensorDataType TDataType, typename T>
        static constexpr auto make_native_value( const T& value ) {
            using NativeType = native_type<TDataType>;
            static_assert(!std::is_void_v<NativeType>, "Unsupported tensor data type for Vulkan");

            if constexpr ( TDataType == TensorDataType::FP16 ) {
                // Special handling for half precision - convert to packed format
                // This would require proper IEEE 754 half-precision conversion
                return static_cast<NativeType>(value); // Simplified for now
            }
            else {
                return static_cast<NativeType>(value);
            }
        }

        /**
         * @brief Converts native Vulkan type values to float for host operations
         *
         * Provides conversion from Vulkan native types back to float values for
         * host-side processing, validation, and cross-platform compatibility.
         * Handles special cases like half-precision and 64-bit types that require
         * explicit device extension support.
         *
         * @tparam TDataType Source abstract tensor data type
         * @param value Native Vulkan type value to convert
         * @return Float representation of the value
         */
        template<TensorDataType TDataType>
        static constexpr float to_float( const native_type<TDataType>& value ) {
            if constexpr ( TDataType == TensorDataType::FP32 ) {
                return value;
            }
            else if constexpr ( TDataType == TensorDataType::FP16 ) {
                // Requires proper IEEE 754 half-precision to float conversion
                // This is a simplified implementation
                return static_cast<float>(value);
            }
            else if constexpr ( TDataType == TensorDataType::FP64 ) {
                return static_cast<float>(value);
            }
            else if constexpr ( TDataType == TensorDataType::INT8 ||
                TDataType == TensorDataType::INT16 ||
                TDataType == TensorDataType::INT32 ||
                TDataType == TensorDataType::INT64 ||
                TDataType == TensorDataType::UINT8 ||
                TDataType == TensorDataType::UINT16 ||
                TDataType == TensorDataType::UINT32 ||
                TDataType == TensorDataType::UINT64 ) {
                return static_cast<float>(value);
            }
            else {
                static_assert(TDataType == TensorDataType::FP32,
                    "Unsupported type conversion to float for Vulkan");
                return 0.0f;
            }
        }

        /**
         * @brief Checks if the specified data type requires device-only access
         *
         * Determines whether a tensor data type can only be accessed from Vulkan
         * compute shaders and cannot be directly manipulated from host code without
         * explicit staging buffer transfers.
         *
         * @tparam TDataType Abstract tensor data type to check
         * @return true if device-only access required, false if host-accessible
         */
        template<TensorDataType TDataType>
        static consteval bool is_device_only() {
            // In Vulkan, FP16 typically requires device-only access and special extensions
            // FP64 also requires specific device extension support
            return TDataType == TensorDataType::FP16 ||
                TDataType == TensorDataType::FP64;
        }

        /**
         * @brief Checks if the data type requires specific Vulkan device extensions
         *
         * Determines whether a tensor data type requires explicit Vulkan device
         * extensions to be enabled, affecting device selection and capability validation.
         *
         * @tparam TDataType Abstract tensor data type to check
         * @return true if extensions required, false for core Vulkan types
         */
        template<TensorDataType TDataType>
        static consteval bool requires_extensions() {
            return TDataType == TensorDataType::FP16 ||  // VK_KHR_shader_float16_int8
                TDataType == TensorDataType::FP64 ||  // VK_KHR_shader_float64
                TDataType == TensorDataType::INT8 ||  // VK_KHR_shader_float16_int8
                TDataType == TensorDataType::INT64 || // VK_KHR_shader_int64
                TDataType == TensorDataType::UINT64;  // VK_KHR_shader_int64
        }

        /**
         * @brief Gets the required Vulkan extension name for the data type
         *
         * Returns the Vulkan extension string required for the specified data type,
         * enabling proper device capability validation and extension enablement.
         *
         * @tparam TDataType Abstract tensor data type requiring extensions
         * @return String view of the required extension name
         */
        template<TensorDataType TDataType>
        static consteval std::string_view required_extension() {
            if constexpr ( TDataType == TensorDataType::FP16 || TDataType == TensorDataType::INT8 ) {
                return "VK_KHR_shader_float16_int8";
            }
            else if constexpr ( TDataType == TensorDataType::FP64 ) {
                return "VK_KHR_shader_float64";
            }
            else if constexpr ( TDataType == TensorDataType::INT64 || TDataType == TensorDataType::UINT64 ) {
                return "VK_KHR_shader_int64";
            }
            else {
                return ""; // No extension required for core types
            }
        }

        /**
         * @brief Gets the SPIR-V OpTypeFloat/OpTypeInt bit width for the data type
         *
         * Returns the bit width used in SPIR-V instruction generation for the
         * specified data type, essential for compute shader compilation and validation.
         *
         * @tparam TDataType Abstract tensor data type
         * @return Bit width for SPIR-V type instructions
         */
        template<TensorDataType TDataType>
        static consteval uint32_t spirv_bit_width() {
            if constexpr ( TDataType == TensorDataType::FP32 ) return 32;
            else if constexpr ( TDataType == TensorDataType::FP16 ) return 16;
            else if constexpr ( TDataType == TensorDataType::FP64 ) return 64;
            else if constexpr ( TDataType == TensorDataType::INT8 || TDataType == TensorDataType::UINT8 ) return 8;
            else if constexpr ( TDataType == TensorDataType::INT16 || TDataType == TensorDataType::UINT16 ) return 16;
            else if constexpr ( TDataType == TensorDataType::INT32 || TDataType == TensorDataType::UINT32 ) return 32;
            else if constexpr ( TDataType == TensorDataType::INT64 || TDataType == TensorDataType::UINT64 ) return 64;
            else {
                static_assert(TDataType == TensorDataType::FP32, "Unsupported data type for SPIR-V bit width");
                return 32;
            }
        }

        /**
         * @brief Gets the Vulkan device type string for debugging
         *
         * @return String identifier for Vulkan backend
         */
        static constexpr std::string_view device_type_name() {
            return "Vulkan";
        }

        /**
         * @brief Gets the preferred memory allocation strategy for the data type
         *
         * Returns the optimal Vulkan memory allocation strategy based on data type
         * characteristics, device capabilities, and expected usage patterns.
         *
         * @tparam TDataType Abstract tensor data type
         * @return Recommended allocation strategy for optimal performance
         */
        template<TensorDataType TDataType>
        static consteval std::string_view preferred_allocation_strategy() {
            if constexpr ( is_device_only<TDataType>() ) {
                return "DEVICE_LOCAL";
            }
            else {
                return "STAGING_OPTIMAL";
            }
        }
    };
}