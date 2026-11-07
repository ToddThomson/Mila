/**
 * @file OperationTraits.Metal.ixx
 * @brief OperationTraits specializations for the Metal backend.
 *
 * FP16 and BF16 only: no FP32 model path exists on Metal (MetalBackend.md section 3.2), so an FP32 row is a compile error.
 */
export module Compute.OperationTraits:Metal;

import Compute.OperationTraits.Template;
import Compute.MetalCausalConv1dOp;
import Compute.MetalElementwiseActivationOp;
import Compute.MetalGatedDeltaRuleOp;
import Compute.MetalGegluOp;
import Compute.MetalGqaOp;
import Compute.MetalLinearOp;
import Compute.MetalMoeOp;
import Compute.MetalNextTokenLogProbabilityOp;
import Compute.MetalResidualOp;
import Compute.MetalRmsNormOp;
import Compute.MetalRopeOp;
import Compute.MetalRouterOp;
import Compute.MetalSamplingOp;
import Compute.MetalSwigluOp;
import Compute.MetalTokenEmbeddingOp;
import Dnn.Quantization.KvCache.PerTokenKvFp8;
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

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TPrecision, PerTokenKvFp8<>>
    {
        using type = Metal::Gqa::MetalGqaOp<TPrecision, PerTokenKvFp8<>>;
    };

    // The sliding window's ring Gemma's local layers hold.
    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TPrecision, SlidingWindowKvCache>
    {
        using type = Metal::Gqa::MetalGqaOp<TPrecision, SlidingWindowKvCache>;
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

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerChannelFp8<>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerChannelFp8<>>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupFp4<128>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupFp4<128>>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupFp4<64>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupFp4<64>>;
    };

    // Q4_0, Gemma's body, and six bits per 32, its quantized table's tied head.
    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupInt4<32>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupInt4<32>>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupInt6<32>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupInt6<32>>;
    };

    // Qwen's two- and three-bit codebooks, at the group sizes CUDA carries.
    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupCodebook2<32>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupCodebook2<32>>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupCodebook2<64>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupCodebook2<64>>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupCodebook3<64>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupCodebook3<64>>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::LinearOp, DeviceType::Metal, TPrecision, PerGroupCodebook3<128>>
    {
        using type = Metal::Linear::MetalLinearOp<TPrecision, PerGroupCodebook3<128>>;
    };

    // The op template on the activation's functor, as CUDA's.
    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::ElementwiseActivationOp, DeviceType::Metal, TPrecision, void>
    {
        template<typename TFunctor>
        using op_for = Metal::Activation::MetalElementwiseActivationOp<TPrecision, TFunctor>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::CausalConv1dOp, DeviceType::Metal, TPrecision, void>
    {
        using type = Metal::Convolution::MetalCausalConv1dOp<TPrecision>;
    };

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::GatedDeltaRuleOp, DeviceType::Metal, TPrecision, void>
    {
        using type = Metal::DeltaNet::MetalGatedDeltaRuleOp<TPrecision>;
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

    template<TensorDataType TPrecision>
        requires ( TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16 )
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Metal, TPrecision, PerGroupInt6<32>>
    {
        using type = Metal::TokenEmbedding::MetalTokenEmbeddingOp<TPrecision, PerGroupInt6<32>>;
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
    struct OperationTraits<OperationType::GegluOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::Geglu::MetalGegluOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::GegluOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::Geglu::MetalGegluOp<TensorDataType::BF16>;
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

    template<>
    struct OperationTraits<OperationType::RouterOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        using type = Metal::Routing::MetalRouterOp<TensorDataType::FP16>;
    };

    template<>
    struct OperationTraits<OperationType::RouterOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        using type = Metal::Routing::MetalRouterOp<TensorDataType::BF16>;
    };

    // The op template on the gate functor, as CUDA's.
    template<>
    struct OperationTraits<OperationType::MoeOp, DeviceType::Metal, TensorDataType::FP16, void>
    {
        template<typename TFunctor, typename TWeightQuantization = NoWeightQuant>
        using op_for = Metal::Moe::MetalMoeOp<TensorDataType::FP16, TFunctor, TWeightQuantization>;
    };

    template<>
    struct OperationTraits<OperationType::MoeOp, DeviceType::Metal, TensorDataType::BF16, void>
    {
        template<typename TFunctor, typename TWeightQuantization = NoWeightQuant>
        using op_for = Metal::Moe::MetalMoeOp<TensorDataType::BF16, TFunctor, TWeightQuantization>;
    };
}
