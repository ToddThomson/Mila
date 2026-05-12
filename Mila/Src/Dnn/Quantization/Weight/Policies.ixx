// Mila - Dnn/Quantization/WeightQuant/Policies.ixx
// Weight quantization policy structs and concept.
//
// Consumers: Linear component and CudaLinearOp only.
// Neither KvCache nor any other subsystem imports this header.

module;
#include <concepts>

export module Dnn.Quantization.Weight.Policies;

import Dnn.TensorDataType;

namespace Mila::Dnn::Quant::Weight
{
    // -------------------------------------------------------------------------
    // NoWeightQuant
    //
    // Identity policy — no quantization. Zero storage overhead, zero runtime
    // cost. All if constexpr branches on kIsQuantized compile away entirely.
    // Default for all Linear instantiations.
    // -------------------------------------------------------------------------
    export struct NoWeightQuant
    {
        static constexpr bool kIsQuantized = false;
        // FP32 is a harmless sentinel — these fields are never read by Linear
        // on the unquantized path; all consumers guard with if constexpr (kIsQuantized).
        static constexpr TensorDataType kStorageDtype = TensorDataType::FP32;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;
        static constexpr bool kPerChannel = false;
    };

    // -------------------------------------------------------------------------
    // PerChannelFp8<TStorage>
    //
    // Per-output-channel FP8 weight quantization. Weights are quantized once at
    // load time (BF16 → FP8_E4M3). One float32 scale per output channel,
    // computed as max(abs(W[o,:])) / 448.0f. Scales are uploaded to device and
    // held for the lifetime of the model. cuBLASLt FP8 matmul consumes weights
    // and scales natively — no dequantization on the forward hot path.
    //
    // TStorage defaults to FP8_E4M3 (higher mantissa precision; correct for
    // stored weights). FP8_E5M2 is reserved for gradients and is not a Mila
    // inference target.
    // -------------------------------------------------------------------------
    export template<TensorDataType TStorage = TensorDataType::FP8_E4M3>
        struct PerChannelFp8
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TStorage;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;
        static constexpr bool kPerChannel = true;
    };

    // -------------------------------------------------------------------------
    // WeightQuantPolicy concept
    //
    // Any type satisfying this concept may be used as the TWeightQuant parameter
    // on Linear and CudaLinearOp. The concept is intentionally narrow — only the
    // fields that Linear needs to make compile-time decisions are required.
    //
    // NoWeightQuant satisfies this concept (kStorageDtype = FP32 sentinel is never
    // read; Linear guards all dtype-consuming paths with if constexpr (kIsQuantized)).
    // -------------------------------------------------------------------------
    export template<typename T>
        concept WeightQuantPolicy = requires
    {
        { T::kIsQuantized  } -> std::convertible_to<bool>;
        {
            T::kStorageDtype
        } -> std::convertible_to<TensorDataType>;
        {
            T::kScaleDtype
        } -> std::convertible_to<TensorDataType>;
        {
            T::kPerChannel
        } -> std::convertible_to<bool>;
    };

    // Verify both concrete policies satisfy the concept at definition time.
    static_assert(WeightQuantPolicy<NoWeightQuant>);
    static_assert(WeightQuantPolicy<PerChannelFp8<>>);

}  // namespace Mila::Dnn::Quant::Weight
