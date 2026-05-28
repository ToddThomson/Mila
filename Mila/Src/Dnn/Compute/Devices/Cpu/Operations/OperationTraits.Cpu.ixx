/**
 * @file OperationTraits.Cpu.ixx
 * @brief OperationTraits specializations for all CPU operation backends.
 *
 * This partition module is the single registration point for every
 * (OperationType, Cpu, TPrecision, TPolicy) -> concrete op mapping.
 *
 * CPU ops are currently concrete (non-templated) FP32-only implementations.
 * BF16 CPU paths are not a current Mila target.
 *
 * Migration status:
 *   LinearOp              pending (CpuLinearOpTypeMap still active)
 *   GeluOp                complete
 *   ResidualOp            complete
 *   SoftmaxOp             complete
 *   MultiHeadAttentionOp  complete
 *   LpeOp                 complete
 *   CrossEntropyOp        pending (CpuSoftmaxCrossEntropyOp not yet wired into CMake)
 *   SamplingOp            pending
 */
export module Compute.OperationTraits:Cpu;

import Compute.OperationTraits.Template;
import Compute.CpuGeluOp;
import Compute.CpuResidualOp;
import Compute.CpuSoftmaxOp;
import Compute.CpuAttention;
import Compute.CpuEncoderOp;

namespace Mila::Dnn::Compute
{
    // -------------------------------------------------------------------------
    // GeluOp — CPU specialization (FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::GeluOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuGeluOp;
    };

    // -------------------------------------------------------------------------
    // ResidualOp — CPU specialization (FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::ResidualOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuResidualOp;
    };

    // -------------------------------------------------------------------------
    // SoftmaxOp — CPU specialization (FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::SoftmaxOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuSoftmaxOp;
    };

    // -------------------------------------------------------------------------
    // MultiHeadAttentionOp — CPU specialization (FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::MultiHeadAttentionOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuAttentionOp;
    };

    // -------------------------------------------------------------------------
    // LpeOp — CPU specialization (INT32 → FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::LpeOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuEncoderOp;
    };

}  // namespace Mila::Dnn::Compute
