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
 *   GroupedQueryAttentionOp  pending (importing CudaGqaOp here creates a module dependency cycle — needs architectural resolution)
 *   SamplingOp               pending
 *   policy-free ops          pending (Softmax, RmsNorm, RoPE, Residual, ...)
 */
export module Compute.OperationTraits:Cuda;

import Compute.OperationTraits.Template;
import Compute.CudaLinearOp;
import Compute.CudaGqaOp;
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

}  // namespace Mila::Dnn::Compute
