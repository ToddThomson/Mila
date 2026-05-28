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
 *   GroupedQueryAttentionOp  complete (NoKvCompression; PerChannelKvFp8 pending CudaGqaOp support)
 *   SamplingOp               pending
 *   policy-free ops          complete
 */
export module Compute.OperationTraits:Cuda;

import Compute.OperationTraits.Template;
import Compute.CudaLinearOp;
import Compute.CudaGqaOp;
import Compute.CudaGeluOp;
import Compute.CudaResidualOp;
import Compute.CudaRmsNormOp;
import Compute.CudaSoftmaxOp;
import Compute.CudaSwigluOp;
import Compute.CudaMultiHeadAttentionOp;
import Compute.CudaRopeOp;
import Compute.CudaLpeOp;
import Compute.CudaTokenEmbeddingOp;
import Compute.CudaSoftmaxCrossEntropyOp;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;
    using namespace Mila::Dnn::Compute::Cuda::Linear;
    using namespace Mila::Dnn::Compute::Cuda::Gqa;

    // -------------------------------------------------------------------------
    // LinearOp — CUDA specializations
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
    // GroupedQueryAttentionOp — CUDA specializations
    //
    // TPolicy = NoKvCompression: uncompressed BF16/FP32 KV cache.
    // TPolicy = PerChannelKvFp8<>: pending CudaGqaOp FP8 cache support.
    // -------------------------------------------------------------------------

    /// Unquantized FP32 path. No KV cache compression.
    template<>
    struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda, TensorDataType::FP32, NoKvCompression>
    {
        using type = CudaGqaOp<TensorDataType::FP32>;
    };

    /// Unquantized BF16 path. No KV cache compression. Standard inference precision.
    template<>
        struct OperationTraits<OperationType::GroupedQueryAttentionOp, DeviceType::Cuda, TensorDataType::BF16, NoKvCompression>
    {
        using type = CudaGqaOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // GeluOp — CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::GeluOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Gelu::CudaGeluOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::GeluOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Gelu::CudaGeluOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // ResidualOp — CUDA specializations
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
    // RmsNormOp — CUDA specializations
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
    // SoftmaxOp — CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::SoftmaxOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Softmax::CudaSoftmaxOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::SoftmaxOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Softmax::CudaSoftmaxOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // SwigluOp — CUDA specializations
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
    // MultiHeadAttentionOp — CUDA specializations
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::MultiHeadAttentionOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::MultiHeadAttention::CudaMultiHeadAttentionOp<TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::MultiHeadAttentionOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::MultiHeadAttention::CudaMultiHeadAttentionOp<TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // RopeOp — CUDA specializations
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
    // LpeOp — CUDA specializations
    // Index type is always INT32 (token position indices).
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::LpeOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::Lpe::CudaLpeOp<TensorDataType::INT32, TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::LpeOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::Lpe::CudaLpeOp<TensorDataType::INT32, TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // TokenEmbeddingOp — CUDA specializations
    // Index type is always INT32 (vocabulary token indices).
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Cuda, TensorDataType::FP32, void>
    {
        using type = Cuda::TokenEmbedding::CudaTokenEmbeddingOp<TensorDataType::INT32, TensorDataType::FP32>;
    };

    template<>
    struct OperationTraits<OperationType::TokenEmbeddingOp, DeviceType::Cuda, TensorDataType::BF16, void>
    {
        using type = Cuda::TokenEmbedding::CudaTokenEmbeddingOp<TensorDataType::INT32, TensorDataType::BF16>;
    };

    // -------------------------------------------------------------------------
    // CrossEntropyOp — CUDA specializations
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

}  // namespace Mila::Dnn::Compute
