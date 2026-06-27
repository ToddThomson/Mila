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
 *   LinearOp              complete (NoWeightQuant; quantized policies are CUDA-only)
 *   GeluOp                complete
 *   LayerNormOp           complete
 *   ResidualOp            complete
 *   SoftmaxOp             complete
 *   MultiHeadAttentionOp  complete
 *   LpeOp                 complete
 *   CrossEntropyOp        pending (CpuSoftmaxCrossEntropyOp not yet wired into CMake)
 *   SamplingOp            pending
 */
export module Compute.OperationTraits:Cpu;

import Compute.OperationTraits.Template;
import Compute.CpuLinearOp;
import Compute.CpuGeluOp;
import Compute.CpuElementwiseActivationOp;
import Compute.CpuLayerNormOp;
import Compute.CpuResidualOp;
import Compute.CpuSoftmaxOp;
import Compute.CpuAttention;
import Compute.CpuEncoderOp;
import Compute.CpuSamplingOp;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn::Quant::Weight;

    // -------------------------------------------------------------------------
    // LinearOp — CPU specialization (FP32, unquantized only)
    //
    // FP32 is the sole CPU-supported precision and CpuLinearOp is concrete
    // (non-templated). Quantized weight policies (PerChannelFp8/PerGroupFp4) are
    // CUDA-only, so NoWeightQuant is the only CPU LinearOp registration.
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::LinearOp, DeviceType::Cpu, TensorDataType::FP32, NoWeightQuant>
    {
        using type = CpuLinearOp;
    };

    // -------------------------------------------------------------------------
    // GeluOp — CPU specialization (FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::GeluOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuGeluOp;
    };

    // -------------------------------------------------------------------------
    // ElementwiseActivationOp — CPU specialization (FP32 only)
    //
    // Unlike concrete CPU ops, this resolves the op *template*: the Activation
    // component maps its compile-time ActivationType to a functor and instantiates
    // op_for<Functor>. No fifth traits axis (see FfnAndMoE.md section 5.1).
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::ElementwiseActivationOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        template<typename TFunctor>
        using op_for = CpuElementwiseActivationOp<TFunctor>;
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
    // LayerNormOp — CPU specialization (FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::LayerNormOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuLayerNormOp;
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

    // -------------------------------------------------------------------------
    // SamplingOp — CPU specialization (FP32 only)
    // -------------------------------------------------------------------------

    template<>
    struct OperationTraits<OperationType::SamplingOp, DeviceType::Cpu, TensorDataType::FP32, void>
    {
        using type = CpuSamplingOp;
    };

}  // namespace Mila::Dnn::Compute
