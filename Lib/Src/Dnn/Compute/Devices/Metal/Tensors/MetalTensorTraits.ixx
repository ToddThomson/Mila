/**
 * @file MetalTensorTraits.ixx
 * @brief Metal-specific tensor trait specializations
 */
module;
#include <cstdint>
#include <string_view>

// Platform-conditional Metal headers
#if defined(__APPLE__) && defined(__has_include)
    #if __has_include(<Metal/Metal.h>)
        #include <Metal/Metal.h>
        #include <simd/simd.h>
        #define METAL_AVAILABLE 1
    #endif
#endif

export module Compute.MetalTensorDataTypeMap;

import Dnn.TensorDataTypeMap;
import Dnn.TensorDataType;
import Compute.DeviceType;

namespace Mila::Dnn
{
    // FUTURE: Metal native types

#ifdef METAL_AVAILABLE
    
    // Metal native types (only on Apple platforms)
    
    template <>
    struct TensorTrait<simd_half> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = true;
        static constexpr DeviceType required_device_type = DeviceType::Metal;
        static constexpr std::string_view type_name = "METAL_FP16";
        static constexpr size_t size_in_bytes = sizeof( simd_half );
        static constexpr TensorDataType data_type = TensorDataType::FP16;
    };

    template <>
    struct TensorTrait<simd_float> {
        static constexpr bool is_float_type = true;
        static constexpr bool is_integer_type = false;
        static constexpr bool is_device_only = false;
        static constexpr DeviceType required_device_type = DeviceType::Metal;
        static constexpr std::string_view type_name = "METAL_FP32";
        static constexpr size_t size_in_bytes = sizeof( simd_float );
        static constexpr TensorDataType data_type = TensorDataType::FP32;
    };
#endif // METAL_AVAILABLE
}