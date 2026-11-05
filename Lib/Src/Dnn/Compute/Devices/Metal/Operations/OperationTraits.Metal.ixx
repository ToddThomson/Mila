/**
 * @file OperationTraits.Metal.ixx
 * @brief OperationTraits specializations for the Metal backend.
 *
 * FP16 and BF16 only: no FP32 model path exists on Metal (MetalBackend.md section 3.2), so an FP32 row is a compile error.
 */
export module Compute.OperationTraits:Metal;

import Compute.OperationTraits.Template;
import Compute.MetalGqaOp;
import Compute.MetalLinearOp;
import Compute.MetalNextTokenLogProbabilityOp;
import Compute.MetalResidualOp;
import Compute.MetalRmsNormOp;
import Compute.MetalRopeOp;
import Compute.MetalSamplingOp;
import Compute.MetalSwigluOp;
import Compute.MetalTokenEmbeddingOp;
import Dnn.Quantization.KvCache.Policy;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn::Quant::KvCache;
    using namespace Mila::Dnn::Quant::Weight;

    template<>
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP16, NoKvCompression>
    {
        using type = Metal::Gqa::MetalGqaOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::BF16, NoKvCompression>
    {
        using type = Metal::Gqa::MetalGqaOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TensorDataType::FP16, NoWeightQuant>
    {
        using type = Metal::Linear::MetalLinearOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TensorDataType::BF16, NoWeightQuant>
    {
        using type = Metal::Linear::MetalLinearOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Metal, TensorDataType::FP16, NoWeightQuant>
    {
        using type = Metal::TokenEmbedding::MetalTokenEmbeddingOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Metal, TensorDataType::BF16, NoWeightQuant>
    {
        using type = Metal::TokenEmbedding::MetalTokenEmbeddingOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::RmsNormOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::RmsNorm::MetalRmsNormOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::RmsNormOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::RmsNorm::MetalRmsNormOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::RopeOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::Rope::MetalRopeOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::RopeOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::Rope::MetalRopeOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::SwigluOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::Swiglu::MetalSwigluOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::SwigluOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::Swiglu::MetalSwigluOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::ResidualOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::Residual::MetalResidualOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::ResidualOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::Residual::MetalResidualOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::SamplingOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::Sampling::MetalSamplingOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::SamplingOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::Sampling::MetalSamplingOp<TensorDataType::BF16>;
    };

    template<>
    struct OperationTraits<OperationType::NextTokenLogProbabilityOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::LogLikelihood::MetalNextTokenLogProbabilityOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::NextTokenLogProbabilityOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::LogLikelihood::MetalNextTokenLogProbabilityOp<TensorDataType::BF16>;
    };
}
