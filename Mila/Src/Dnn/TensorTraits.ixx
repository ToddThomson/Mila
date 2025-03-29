module;
#include <vector>
#include <cuda_fp16.h>
#include <type_traits>

export module Dnn.TensorTraits;

import Dnn.TensorType;

namespace Mila::Dnn
{
	template <typename T>
    struct TensorTypeTrait;

    template <>
    struct TensorTypeTrait<float> {
        static constexpr TensorType value = TensorType::FP32;
    };

    template <>
    struct TensorTypeTrait<half> {
        static constexpr TensorType value = TensorType::FP16;
    };

    template <>
    struct TensorTypeTrait<int16_t> {
        static constexpr TensorType value = TensorType::INT16;
    };

    template <>
    struct TensorTypeTrait<int> {
        static constexpr TensorType value = TensorType::INT32;
    };

    template <>
    struct TensorTypeTrait<uint16_t> {
        static constexpr TensorType value = TensorType::UINT16;
    };

    template <>
    struct TensorTypeTrait<uint32_t> {
        static constexpr TensorType value = TensorType::UINT32;
    };


    export template <typename T>
    concept ValidTensorType = requires {
        { TensorTypeTrait<T>::value } -> std::convertible_to<TensorType>;
    };

    export template <typename TInput, typename TCompute>
    concept ValidTensorTypes = ValidTensorType<TInput> && ValidTensorType<TCompute>;
}