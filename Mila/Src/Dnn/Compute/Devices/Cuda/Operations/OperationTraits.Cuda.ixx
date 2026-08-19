/**
 * @file OperationTraits.Cuda.ixx
 * @brief OperationTraits specializations for all CUDA operation backends.
 *
 * This partition module is the single registration point for every
 * (OperationType, Cuda, TPrecision, TPolicy) -> concrete op mapping.
 * Add a new specialization block here when migrating a component from its
 * legacy *OpTypeMap to the unified OperationTraits dispatch.
 *
 * Migration status:
 *   LinearOp                 complete
 *   GroupedQueryAttentionOp  complete (NoKvCompression, SlidingWindowKvCache; PerChannelKvFp8 pending CudaGqaOp support)
 *   SamplingOp               pending
 *   policy-free ops          complete
 */
export module Compute.OperationTraits:Cuda;

import Compute.OperationTraits.Template;
import Compute.CudaLinearOp;
import Compute.CudaGqaOp;
import Compute.CudaGeluOp;
import Compute.CudaElementwiseActivationOp;
import Compute.CudaResidualOp;
import Compute.CudaRmsNormOp;
import Compute.CudaLayerNormOp;
import Compute.CudaSoftmaxOp;
import Compute.CudaSwigluOp;
import Compute.CudaGegluOp;
import Compute.CudaMultiHeadAttentionOp;
import Compute.CudaRopeOp;
import Compute.CudaLpeOp;
import Compute.CudaTokenEmbeddingOp;
import Compute.CudaSoftmaxCrossEntropyOp;
import Compute.CudaSamplingOp;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;
    using namespace Mila::Dnn::Compute::Cuda::Linear;
    using namespace Mila::Dnn::Compute::Cuda::Gqa;
    using namespace Mila::Dnn::Compute::Cuda::Sampling;

    // -------------------------------------------------------------------------
    // LinearOp -- CUDA specializations
    // -------------------------------------------------------------------------

    /// Unquantized FP32 path. Retained for validation and reference.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::FP32, NoWeightQuant>
    {
        using type = CudaLinearOp<TensorDataType::FP32, NoWeightQuant>;
    };

    /// Unquantized BF16 path. Standard inference precision.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, NoWeightQuant>
    {
        using type = CudaLinearOp<TensorDataType::BF16, NoWeightQuant>;
    };

    /// FP8 per-channel quantized BF16 path. Requires SM >= 8.0 (Ampere+).
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerChannelFp8<>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerChannelFp8<>>;
    };

    // Sub-4-bit codebook paths (Specifications/Qwen3.8.md). BF16 only: the GEMV kernels
    // are BF16 in / BF16 out, and an op that cannot honor a precision must carry no row
    // for it. They resolve to CudaLinearOp like every other weight format -- the policy,
    // not a second operation, is what selects the decode kernel and the prefill strategy.

    /// W2A16, 2-bit codes into a 4-entry table, group 32. Section 5 FFN gate/up.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupCodebook2<32>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupCodebook2<32>>;
    };

    /// W2A16 at group 64. Half the scale overhead, coarser absmax.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupCodebook2<64>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupCodebook2<64>>;
    };

    /// W3A16, 3-bit codes into an 8-entry table, group 64. Section 5 FFN down.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupCodebook3<64>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupCodebook3<64>>;
    };

    /// W3A16 at group 128.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupCodebook3<128>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupCodebook3<128>>;
    };

    /// INT4 per-group quantized BF16 path. W4A16 fused GEMM, group_size=128. Requires SM >= 8.0.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupInt4<128>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupInt4<128>>;
    };

    /// INT4 per-group quantized BF16 path. W4A16 fused GEMM, group_size=64. Requires SM >= 8.0.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupInt4<64>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupInt4<64>>;
    };

    /// FP4 E2M1 per-group quantized BF16 path. W4A16 fused GEMM with E2M1 decode, group_size=128. Requires SM >= 8.0.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupFp4<128>>;
    };

    /// FP4 E2M1 per-group quantized BF16 path. W4A16 fused GEMM with E2M1 decode, group_size=64. Requires SM >= 8.0.
    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<64>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerGroupFp4<64>>;
    };

    // -------------------------------------------------------------------------
    // GroupedQueryAttentionOp -- CUDA specializations
    //
    // TPolicy = NoKvCompression:      uncompressed full-context BF16/FP32 KV cache.
    // TPolicy = SlidingWindowKvCache: uncompressed bounded ring cache for sliding
    //                                 layers (CudaGqaOp kBounded axis, SlidingWindowKvCache.md).
    // TPolicy = PerChannelKvFp8<>:    pending CudaGqaOp FP8 cache support.
    // -------------------------------------------------------------------------

    /// Unquantized FP32 path. Full-context KV cache.
    template<>
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda, TensorDataType::FP32, NoKvCompression>
    {
        using type = CudaGqaOp<TensorDataType::FP32, false>;
    };

    /// Unquantized BF16 path. Full-context KV cache. Standard inference precision.
    template<>
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda, TensorDataType::BF16, NoKvCompression>
    {
        using type = CudaGqaOp<TensorDataType::BF16, false>;
    };

    /// Bounded sliding-window ring cache, FP32. Sliding (local) layers only (window > 0).
    template<>
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda, TensorDataType::FP32, SlidingWindowKvCache>
    {
        using type = CudaGqaOp<TensorDataType::FP32, true>;
    };

    /// Bounded sliding-window ring cache, BF16. Sliding (local) layers only (window > 0).
    template<>
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda, TensorDataType::BF16, SlidingWindowKvCache>
    {
        using type = CudaGqaOp<TensorDataType::BF16, true>;
    };

    // -------------------------------------------------------------------------
    // GeluOp -- CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::GeluOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Gelu::CudaGeluOp<TensorDataType::FP32>;
    };

    // No BF16 row: cuda_gelu_impl is `float || half` only, so a BF16 row would advertise
    // an op that hard-errors on instantiation. FP32-only is the honest advertisement; the
    // BF16 FFN path uses GegluOp, not the standalone GeluOp. Re-add with a BF16 kernel, not
    // a bare row, if a BF16 GeluOp is ever needed.

    // -------------------------------------------------------------------------
    // ElementwiseActivationOp -- CUDA specializations (FP32, BF16)
    //
    // Resolves the op *template*: the Activation component maps its compile-time
    // ActivationType to a functor and instantiates op_for<Functor>. No fifth traits
    // axis (see FfnAndMoE.md section 5.1).
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::ElementwiseActivationOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        template<typename TFunctor>
        using op_for = Cuda::Activation::CudaElementwiseActivationOp<TensorDataType::FP32, TFunctor>;
    };

    template<>
    struct OperationTraits<OperationType::ElementwiseActivationOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        template<typename TFunctor>
        using op_for = Cuda::Activation::CudaElementwiseActivationOp<TensorDataType::BF16, TFunctor>;
    };

    // -------------------------------------------------------------------------
    // ResidualOp -- CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::ResidualOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Residual::CudaResidualOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::ResidualOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Residual::CudaResidualOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // RmsNormOp -- CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::RmsNormOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::RmsNorm::CudaRmsNormOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::RmsNormOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::RmsNorm::CudaRmsNormOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // LayerNormOp -- CUDA specializations
    //
    // GPT-2 lineage. FP32 and FP16 kernels only (no BF16 LayerNorm kernel); the
    // GPT-2 inference path runs at FP32.
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::LayerNormOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::LayerNorm::CudaLayerNormOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::LayerNormOp, DeviceType::Cuda, TensorDataType::FP16, void>
    {
        using type = Cuda::LayerNorm::CudaLayerNormOp<TensorDataType::FP16>;
    };

    // -------------------------------------------------------------------------
    // SoftmaxOp -- CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::SoftmaxOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Softmax::CudaSoftmaxOp<TensorDataType::FP32>;
    };

    // No BF16 row: cuda_softmax_forward is `float || half` only. The BF16 attention path
    // runs softmax inside the fused GQA/decode kernels, not through the standalone SoftmaxOp.
    // FP32-only is honest here; re-add with a BF16 kernel, not a bare row, if ever needed.

    // -------------------------------------------------------------------------
    // SwigluOp -- CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::SwigluOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Swiglu::CudaSwigluOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::SwigluOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Swiglu::CudaSwigluOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // GegluOp -- CUDA specializations (GELU-gated GLU; Gemma FFN, forward only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::GegluOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Geglu::CudaGegluOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::GegluOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Geglu::CudaGegluOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // MultiHeadAttentionOp -- CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::MultiHeadAttentionOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::MultiHeadAttention::CudaMultiHeadAttentionOp<TensorDataType::FP32>;
    };

    // No BF16 row: CudaMhaOp dispatch is `float || half` only (the GPT-2 lineage runs MHA at
    // FP32). Llama/Gemma use GQA, not MHA. FP32-only is honest; re-add with a BF16 kernel,
    // not a bare row, if a BF16 MHA is ever needed. (Resolves the BF16 REVIEW in CudaMhaOp.Dispatch.ixx.)

    // -------------------------------------------------------------------------
    // RopeOp -- CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::RopeOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Rope::CudaRopeOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::RopeOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Rope::CudaRopeOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // LpeOp -- CUDA specializations
    // Index type is always INT32 (token position indices).
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::LpeOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Lpe::CudaLpeOp<TensorDataType::INT32, TensorDataType::FP32>;
    };

    // No BF16 row: CudaLpeOp dispatch is `float || half` only. Learned positional embeddings
    // are a GPT-2-lineage feature and that path runs at FP32; Llama/Gemma use RoPE. FP32-only
    // is honest; re-add with a BF16 kernel, not a bare row, if ever needed. (Resolves the BF16
    // REVIEW in CudaLpeOp.Dispatch.ixx.)

    // -------------------------------------------------------------------------
    // TokenEmbeddingOp -- CUDA specializations
    // Index type is always INT32 (vocabulary token indices).
    // TPolicy = table quantization policy (D4 Design B): NoWeightQuant keeps the
    // full-precision table; PerChannelFp8<> stores FP8_E4M3 rows + FP32 row scales
    // shared with a tied lm_head.
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Cuda, TensorDataType::FP32, NoWeightQuant>
    {
        using type = Cuda::TokenEmbedding::CudaTokenEmbeddingOp<TensorDataType::INT32, TensorDataType::FP32, NoWeightQuant>;
    };

    template<>
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Cuda, TensorDataType::BF16, NoWeightQuant>
    {
        using type = Cuda::TokenEmbedding::CudaTokenEmbeddingOp<TensorDataType::INT32, TensorDataType::BF16, NoWeightQuant>;
    };

    /// FP8 per-vocab-row quantized table, BF16 gather-dequant output (D4 Design B).
    template<>
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Cuda, TensorDataType::BF16, PerChannelFp8<>>
    {
        using type = Cuda::TokenEmbedding::CudaTokenEmbeddingOp<TensorDataType::INT32, TensorDataType::BF16, PerChannelFp8<>>;
    };

    // -------------------------------------------------------------------------
    // CrossEntropyOp -- CUDA specializations
    // Logits precision matches compute precision; target type is always INT32.
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::CrossEntropyOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::SoftmaxCrossEntropy::CudaSoftmaxCrossEntropyOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::CrossEntropyOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::SoftmaxCrossEntropy::CudaSoftmaxCrossEntropyOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // SamplingOp -- CUDA specializations (logits precision; INT32 token output)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::SamplingOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Sampling::CudaSamplingOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::SamplingOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Sampling::CudaSamplingOp<TensorDataType::BF16>;
    };

}  // namespace Mila::Dnn::Compute
