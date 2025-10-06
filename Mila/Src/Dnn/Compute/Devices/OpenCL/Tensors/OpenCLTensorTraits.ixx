/**
 * @file OpenCLTensorTraits.ixx
 * @brief OpenCL-specific tensor trait specializations for heterogeneous compute
 *
 * This module provides OpenCL-specific tensor characteristics and native type mappings
 * for cross-platform compute operations. Supports various OpenCL data types across
 * different vendors (NVIDIA, AMD, Intel, ARM) and device types (GPU, CPU, FPGA).
 */

module;
#include <cstdint>
#include <string_view>
#include <type_traits>

#ifdef OPENCL_AVAILABLE
#include <CL/cl.h>
#include <CL/cl2.hpp>
#endif

export module Compute.OpenCLTensorTraits;

import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Compute.DeviceType;

namespace Mila::Dnn
{
#ifdef OPENCL_AVAILABLE

    // OpenCL native type specializations for cross-platform compatibility

    template <>
    struct TensorTrait<cl_half> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = true;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_FP16";
        static constexpr size_t size_in_bytes = sizeof( cl_half );
        static constexpr TensorDataType data_type = TensorDataType::FP16;
    };

    template <>
    struct TensorTrait<cl_float> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_FP32";
        static constexpr size_t size_in_bytes = sizeof( cl_float );
        static constexpr TensorDataType data_type = TensorDataType::FP32;
    };

    template <>
    struct TensorTrait<cl_double> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_FP64";
        static constexpr size_t size_in_bytes = sizeof( cl_double );
        static constexpr TensorDataType data_type = TensorDataType::FP64;
    };

    template <>
    struct TensorTrait<cl_char> {
        static constexpr bool is_float_type = false;
        static constexpr bool is_integer_type = true;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_INT8";
        static constexpr size_t size_in_bytes = sizeof( cl_char );
        static constexpr TensorDataType data_type = TensorDataType::INT8;
    };

    template <>
    struct TensorTrait<cl_short> {
        static constexpr bool is_float_type = false;
        static constexpr bool is_integer_type = true;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_INT16";
        static constexpr size_t size_in_bytes = sizeof( cl_short );
        static constexpr TensorDataType data_type = TensorDataType::INT16;
    };

    template <>
    struct TensorTrait<cl_int> {
        static constexpr bool is_float_type = false;
        static constexpr bool is_integer_type = true;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_INT32";
        static constexpr size_t size_in_bytes = sizeof( cl_int );
        static constexpr TensorDataType data_type = TensorDataType::INT32;
    };

    template <>
    struct TensorTrait<cl_uchar> {
        static constexpr bool is_float_type = false;
        static constexpr bool is_integer_type = true;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_UINT8";
        static constexpr size_t size_in_bytes = sizeof( cl_uchar );
        static constexpr TensorDataType data_type = TensorDataType::UINT8;
    };

    template <>
    struct TensorTrait<cl_ushort> {
        static constexpr bool is_float_type = false;
        static constexpr bool is_integer_type = true;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_UINT16";
        static constexpr size_t size_in_bytes = sizeof( cl_ushort );
        static constexpr TensorDataType data_type = TensorDataType::UINT16;
    };

    template <>
    struct TensorTrait<cl_uint> {
        static constexpr bool is_float_type = false;
        static constexpr bool is_integer_type = true;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::OpenCL;
        static constexpr std::string_view type_name = "OPENCL_UINT32";
        static constexpr size_t size_in_bytes = sizeof( cl_uint );
        static constexpr TensorDataType data_type = TensorDataType::UINT32;
    };

#endif // OPENCL_AVAILABLE

    /**
     * @brief OpenCL-specific traits for abstract tensor data types
     *
     * Provides device-specific characteristics and native type mappings for
     * OpenCL-supported tensor data types, enabling type-safe value initialization
     * and cross-platform compute operations across different OpenCL implementations.
     *
     * Supports heterogeneous compute across:
     * - GPU devices (NVIDIA, AMD, Intel, ARM Mali)
     * - CPU devices (Intel, AMD, ARM)
     * - FPGA and accelerator devices
     * - Custom OpenCL implementations
     */
    export class OpenCLTensorTraits {
    public:
        /**
         * @brief Validates OpenCL support for abstract tensor data types
         *
         * Determines whether OpenCL devices and memory resources support the specified
         * abstract tensor data type, enabling compile-time validation across different
         * OpenCL platforms and implementations.
         *
         * @tparam TDataType Abstract tensor data type to validate
         * @return true if OpenCL supports the data type, false otherwise
         */
        template<TensorDataType TDataType>
        static consteval bool supports() {
            return TDataType == TensorDataType::FP32 ||
                TDataType == TensorDataType::FP16 ||
                TDataType == TensorDataType::FP64 ||
                TDataType == TensorDataType::INT8 ||
                TDataType == TensorDataType::INT16 ||
                TDataType == TensorDataType::INT32 ||
                TDataType == TensorDataType::UINT8 ||
                TDataType == TensorDataType::UINT16 ||
                TDataType == TensorDataType::UINT32;
        }

    private:
        /**
         * @brief Helper function for compile-time native type deduction
         *
         * Maps abstract tensor data types to OpenCL native types, avoiding
         * circular template dependencies during instantiation. Handles both
         * available and unavailable OpenCL configurations.
         */
        template<TensorDataType TDataType>
        static consteval auto get_native_type() {
        #ifdef OPENCL_AVAILABLE
            if constexpr ( TDataType == TensorDataType::FP32 ) return cl_float{};
            else if constexpr ( TDataType == TensorDataType::FP16 ) return cl_half{};
            else if constexpr ( TDataType == TensorDataType::FP64 ) return cl_double{};
            else if constexpr ( TDataType == TensorDataType::INT8 ) return cl_char{};
            else if constexpr ( TDataType == TensorDataType::INT16 ) return cl_short{};
            else if constexpr ( TDataType == TensorDataType::INT32 ) return cl_int{};
            else if constexpr ( TDataType == TensorDataType::UINT8 ) return cl_uchar{};
            else if constexpr ( TDataType == TensorDataType::UINT16 ) return cl_ushort{};
            else if constexpr ( TDataType == TensorDataType::UINT32 ) return cl_uint{};
            else {
                static_assert(TDataType == TensorDataType::FP32,
                    "Unsupported tensor data type for OpenCL. Check OpenCLTensorTraits::supports() for supported types.");
                return cl_float{};
            }
        #else
            // When OpenCL is not available, provide host-compatible fallback types
            if constexpr ( TDataType == TensorDataType::FP32 ) return float{};
            else if constexpr ( TDataType == TensorDataType::FP16 ) return float{};  // No native half support
            else if constexpr ( TDataType == TensorDataType::FP64 ) return double{};
            else if constexpr ( TDataType == TensorDataType::INT8 ) return int8_t{};
            else if constexpr ( TDataType == TensorDataType::INT16 ) return int16_t{};
            else if constexpr ( TDataType == TensorDataType::INT32 ) return int32_t{};
            else if constexpr ( TDataType == TensorDataType::UINT8 ) return uint8_t{};
            else if constexpr ( TDataType == TensorDataType::UINT16 ) return uint16_t{};
            else if constexpr ( TDataType == TensorDataType::UINT32 ) return uint32_t{};
            else {
                static_assert(TDataType == TensorDataType::FP32,
                    "OpenCL not available - tensor data type not supported in fallback mode.");
                return float{};
            }
        #endif
        }

    public:
        /**
         * @brief Type alias for the native OpenCL type corresponding to a TensorDataType
         *
         * Provides compile-time mapping from abstract TensorDataType enumeration
         * to concrete OpenCL types, ensuring type safety across different OpenCL
         * implementations and platforms.
         */
        template<TensorDataType TDataType>
        using native_type = decltype(get_native_type<TDataType>());

        /**
         * @brief Creates a native type value from a compatible input value
         *
         * Provides safe conversion from input values to the appropriate native
         * OpenCL type, handling cross-platform type conversions and ensuring
         * compatibility across different OpenCL implementations.
         *
         * @tparam TDataType Target abstract tensor data type
         * @tparam T Input value type
         * @param value Input value to convert
         * @return Native OpenCL type value suitable for compute operations
         */
        template<TensorDataType TDataType, typename T>
        static constexpr auto make_native_value( const T& value ) {
            using NativeType = native_type<TDataType>;
            static_assert(!std::is_void_v<NativeType>, "Unsupported tensor data type for OpenCL");
            return static_cast<NativeType>(value);
        }

        /**
         * @brief Converts native OpenCL type values to float for host operations
         *
         * Provides conversion from OpenCL native types back to float values for
         * host-side processing, validation, and cross-platform compatibility.
         * Essential for tensor initialization and debugging operations.
         *
         * @tparam TDataType Source abstract tensor data type
         * @param value Native OpenCL type value to convert
         * @return Float representation of the value
         */
        template<TensorDataType TDataType>
        static constexpr float to_float( const native_type<TDataType>& value ) {
        #ifdef OPENCL_AVAILABLE
            if constexpr ( TDataType == TensorDataType::FP32 ) {
                return static_cast<float>(value);
            }
            else if constexpr ( TDataType == TensorDataType::FP16 ) {
                // OpenCL half to float conversion - implementation specific
                return static_cast<float>(value);
            }
            else if constexpr ( TDataType == TensorDataType::FP64 ) {
                return static_cast<float>(value);
            }
            else if constexpr ( TDataType == TensorDataType::INT8 ||
                TDataType == TensorDataType::INT16 ||
                TDataType == TensorDataType::INT32 ||
                TDataType == TensorDataType::UINT8 ||
                TDataType == TensorDataType::UINT16 ||
                TDataType == TensorDataType::UINT32 ) {
                return static_cast<float>(value);
            }
            else {
                static_assert(TDataType == TensorDataType::FP32,
                    "Unsupported type conversion to float for OpenCL");
                return 0.0f;
            }
        #else
            // Fallback conversion when OpenCL is not available
            return static_cast<float>(value);
        #endif
        }

        /**
         * @brief Checks if the specified data type requires device-only access
         *
         * Determines whether a tensor data type can only be accessed from OpenCL
         * device kernels and cannot be directly manipulated from host code.
         *
         * @tparam TDataType Abstract tensor data type to check
         * @return true if device-only access required, false if host-accessible
         */
        template<TensorDataType TDataType>
        static consteval bool is_device_only() {
            // In OpenCL, FP16 is typically device-only on most platforms
            return TDataType == TensorDataType::FP16;
        }

        /**
         * @brief Gets the OpenCL device type string for debugging
         *
         * @return String identifier for OpenCL backend
         */
        static constexpr std::string_view device_type_name() {
            return "OpenCL";
        }
    };
}