/**
 * @file CudaTensorTraits-TensorTraitSpecializations.ixx
 * @brief CUDA concrete type trait specializations
 */
module;
#include <cstdint>
#include <string_view>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

export module Compute.CudaTensorDataType:Specializations;

import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Compute.DeviceType;

namespace Mila::Dnn 
{
    using namespace Compute;

    // CUDA-specific concrete type specializations
    template <>
    struct TensorDataTypeMap<half> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = true;
        static constexpr DeviceType required_device_type = DeviceType::Cuda;
        static constexpr std::string_view type_name = "FP16";
        static constexpr size_t size_in_bytes = sizeof( half );
        static constexpr TensorDataType data_type = TensorDataType::FP16;
    };

    template <>
    struct TensorDataTypeMap<nv_bfloat16> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = true;
        static constexpr DeviceType required_device_type = DeviceType::Cuda;
        static constexpr std::string_view type_name = "BF16";
        static constexpr size_t size_in_bytes = sizeof( nv_bfloat16 );
        static constexpr TensorDataType data_type = TensorDataType::BF16;
    };

    template <>
    struct TensorDataTypeMap<__nv_fp8_e4m3> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = true;
        static constexpr DeviceType required_device_type = DeviceType::Cuda;
        static constexpr std::string_view type_name = "FP8_E4M3";
        static constexpr size_t size_in_bytes = sizeof( __nv_fp8_e4m3 );
        static constexpr TensorDataType data_type = TensorDataType::FP8_E4M3;
    };

    template <>
    struct TensorDataTypeMap<__nv_fp8_e5m2> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = true;
        static constexpr DeviceType required_device_type = DeviceType::Cuda;
        static constexpr std::string_view type_name = "FP8_E5M2";
        static constexpr size_t size_in_bytes = sizeof( __nv_fp8_e5m2 );
        static constexpr TensorDataType data_type = TensorDataType::FP8_E5M2;
    };
}