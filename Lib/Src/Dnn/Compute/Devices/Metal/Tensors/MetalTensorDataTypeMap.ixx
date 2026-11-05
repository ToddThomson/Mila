/**
 * @file MetalTensorDataTypeMap.ixx
 * @brief From a tensor's data type to its element on the host and its name in Metal's kernels.
 */

export module Compute.MetalTensorDataTypeMap;

import std;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute
{
    /// Undefined for a type Metal's kernels do not carry, so using one is a compile error.
    export template<TensorDataType TDataType>
    struct MetalTensorDataTypeMap;

    /**
     * @brief The bytes `count` elements take.
     *
     * What a tensor's own elements take, which a view needs: `Tensor::getStorageSize` is its whole buffer's. A packed
     * format below a byte per element does not compile here, since its count of elements need not start on a byte.
     */
    export template<TensorDataType TDataType>
    [[nodiscard]] constexpr std::size_t metalElementBytes( dim_t count )
    {
        static_assert( !requires { TensorDataTypeTraits<TDataType>::bits_per_element; },
            "A packed weight format has no Metal path yet: MetalBackend.md Phase 5" );

        return static_cast<std::size_t>( count ) * TensorDataTypeTraits<TDataType>::size_in_bytes;
    }

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::FP32>
    {
        using host_type = float;
        static constexpr std::string_view kernel_type = "float";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::FP16>
    {
        using host_type = __fp16;
        static constexpr std::string_view kernel_type = "half";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::BF16>
    {
        using host_type = __bf16;
        static constexpr std::string_view kernel_type = "bfloat";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::INT32>
    {
        using host_type = std::int32_t;
        static constexpr std::string_view kernel_type = "int";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::UINT32>
    {
        using host_type = std::uint32_t;
        static constexpr std::string_view kernel_type = "uint";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::INT16>
    {
        using host_type = std::int16_t;
        static constexpr std::string_view kernel_type = "short";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::UINT16>
    {
        using host_type = std::uint16_t;
        static constexpr std::string_view kernel_type = "ushort";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::INT8>
    {
        using host_type = std::int8_t;
        static constexpr std::string_view kernel_type = "char";
    };

    template<>
    struct MetalTensorDataTypeMap<TensorDataType::UINT8>
    {
        using host_type = std::uint8_t;
        static constexpr std::string_view kernel_type = "uchar";
    };
}
