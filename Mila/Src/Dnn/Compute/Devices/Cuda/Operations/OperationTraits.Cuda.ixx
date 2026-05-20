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
 *   LinearOp            complete
 *   GroupedQueryAttentionOp  pending
 *   SamplingOp               pending
 *   policy-free ops          pending (Softmax, RmsNorm, RoPE, Residual, ...)
 */
export module Compute.OperationTraits:Cuda;

import Compute.OperationTraits.Template;
import Compute.CudaLinearOp;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Compute::Cuda::Linear;

    // -------------------------------------------------------------------------
    // LinearOp — CUDA specializations
    // -------------------------------------------------------------------------

    /// Unquantized FP32 path. Retained for validation and reference.
    export template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::FP32, NoWeightQuant>
    {
        using type = CudaLinearOp<TensorDataType::FP32, NoWeightQuant>;
    };

    /// Unquantized BF16 path. Standard inference precision.
    export template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, NoWeightQuant>
    {
        using type = CudaLinearOp<TensorDataType::BF16, NoWeightQuant>;
    };

    /// FP8 per-channel quantized BF16 path. Requires SM >= 8.9.
    export template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, PerChannelFp8<>>
    {
        using type = CudaLinearOp<TensorDataType::BF16, PerChannelFp8<>>;
    };

}  // namespace Mila::Dnn::Compute
