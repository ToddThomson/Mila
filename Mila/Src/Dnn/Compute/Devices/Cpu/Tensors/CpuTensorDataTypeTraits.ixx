/**
 * @file CpuTensorDataTypeTraits.ixx
 * @brief CPU-specific tensor trait specializations
 */
module;
#include <type_traits>

export module Compute.CpuTensorDataTypeTraits;

import Dnn.TensorDataType;

namespace Mila::Dnn
{
    /**
     * @brief CPU-specific traits for abstract tensor data types
     *
     * Provides device-specific characteristics and native type mappings for
     * CPU-supported tensor data types, enabling type-safe value initialization
     * and host-specific optimizations.
     */
    export class CpuTensorDataTypeTraits {
    public:
        /**
         * @brief Validates CPU support for abstract tensor data types
         *
         * Determines whether CPU devices and memory resources support the specified
         * abstract tensor data type. CPU typically supports standard C++ types but
         * not device-specific precision formats.
         *
         * @tparam TDataType Abstract tensor data type to validate
         * @return true if CPU supports the data type, false otherwise
         */
        template<TensorDataType TDataType>
        static consteval bool supports() {
            return TDataType == TensorDataType::FP32 ||
                TDataType == TensorDataType::INT8 ||
                TDataType == TensorDataType::INT16 ||
                TDataType == TensorDataType::INT32 ||
                TDataType == TensorDataType::UINT8 ||
                TDataType == TensorDataType::UINT16 ||
                TDataType == TensorDataType::UINT32;
        }

        /**
         * @brief Maps abstract tensor data types to concrete CPU native types
         *
         * Provides compile-time mapping from abstract TensorDataType enumeration
         * to concrete standard C++ types supported on CPU, enabling type-safe
         * value initialization and host operations.
         *
         * @tparam TDataType Abstract tensor data type to map
         * @return Concrete C++ type corresponding to the abstract type
         *
         * @note Only supports types where supports<TDataType>() returns true
         * @note All mapped types are standard C++ types compatible with host code
         */
        template<TensorDataType TDataType>
        using native_type = 
            std::conditional_t<TDataType == TensorDataType::FP32, float,
            std::conditional_t<TDataType == TensorDataType::INT8, int8_t,
            std::conditional_t<TDataType == TensorDataType::INT16, int16_t,
            std::conditional_t<TDataType == TensorDataType::INT32, int32_t,
            std::conditional_t<TDataType == TensorDataType::UINT8, uint8_t,
            std::conditional_t<TDataType == TensorDataType::UINT16, uint16_t,
            std::conditional_t<TDataType == TensorDataType::UINT32, uint32_t,
            void>>>>>>>;
    };
}