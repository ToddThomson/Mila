/**
 * @file CudaTensorTraits-TypeMaps.ixx
 * @brief CUDA abstract-to-concrete type mapping utilities
 */
module;
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

export module Compute.CudaTensorTraits:Maps;

import Dnn.TensorDataType;

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Maps abstract tensor data types to concrete CUDA native types
     */
    export template<TensorDataType TDataType>
        struct NativeDataTypeMap {
        static_assert(TDataType != TDataType, "Unsupported native data type mapping");
    };

    template<> struct NativeDataTypeMap<TensorDataType::FP32> { using type = float; };
    template<> struct NativeDataTypeMap<TensorDataType::FP16> { using type = __half; };
    template<> struct NativeDataTypeMap<TensorDataType::BF16> { using type = __nv_bfloat16; };
    template<> struct NativeDataTypeMap<TensorDataType::FP8_E4M3> { using type = __nv_fp8_e4m3; };
    template<> struct NativeDataTypeMap<TensorDataType::FP8_E5M2> { using type = __nv_fp8_e5m2; };
    template<> struct NativeDataTypeMap<TensorDataType::INT8> { using type = int8_t; };
    template<> struct NativeDataTypeMap<TensorDataType::INT16> { using type = int16_t; };
    template<> struct NativeDataTypeMap<TensorDataType::INT32> { using type = int32_t; };
    template<> struct NativeDataTypeMap<TensorDataType::UINT8> { using type = uint8_t; };
    template<> struct NativeDataTypeMap<TensorDataType::UINT16> { using type = uint16_t; };
    template<> struct NativeDataTypeMap<TensorDataType::UINT32> { using type = uint32_t; };

    /**
     * @brief Maps abstract tensor data types to concrete host-compatible types
     */
    export template<TensorDataType TDataType>
        struct HostDataTypeMap {
        static_assert(TDataType != TDataType, "Unsupported host data type mapping");
    };

    template<> struct HostDataTypeMap<TensorDataType::FP32> { using type = float; };
    template<> struct HostDataTypeMap<TensorDataType::FP16> { using type = float; };
    template<> struct HostDataTypeMap<TensorDataType::BF16> { using type = float; };
    template<> struct HostDataTypeMap<TensorDataType::FP8_E4M3> { using type = float; };
    template<> struct HostDataTypeMap<TensorDataType::FP8_E5M2> { using type = float; };
    template<> struct HostDataTypeMap<TensorDataType::INT8> { using type = int8_t; };
    template<> struct HostDataTypeMap<TensorDataType::INT16> { using type = int16_t; };
    template<> struct HostDataTypeMap<TensorDataType::INT32> { using type = int32_t; };
    template<> struct HostDataTypeMap<TensorDataType::UINT8> { using type = uint8_t; };
    template<> struct HostDataTypeMap<TensorDataType::UINT16> { using type = uint16_t; };
    template<> struct HostDataTypeMap<TensorDataType::UINT32> { using type = uint32_t; };
}