/**
 * @file OperationTraits.Template.ixx
 * @brief Unified compile-time dispatch template mapping
 *        (OperationType, DeviceType, TPrecision, TPolicy) to a concrete operation type.
 *
 * Replaces all per-component *OpTypeMap templates with a single keyed dispatch table.
 * Backend specializations live in OperationTraits:Cuda and OperationTraits:Cpu partition
 * modules co-located with their concrete op implementations.
 *
 * A missing specialization for any combination is a hard compile error -- the correct
 * diagnostic for an unsupported (OperationType, DeviceType, precision, policy) tuple.
 *
 * Policy conventions:
 *   TPolicy = void                  policy-free ops  (Softmax, RmsNorm, RoPE, Residual, ...)
 *   TPolicy = WeightQuantPolicy     LinearOp         (NoWeightQuant, PerChannelFp8<>, ...)
 *   TPolicy = KvCachePolicy         GQA              (NoKvCompression, PerChannelKvFp8<>, ...)
 *
 * Components hold the concrete op type via a local alias:
 *
 *   using OpType = OperationTraits<OperationType::LinearOp,
 *                                  TDeviceType, TPrecision, TWeightQuant>::type;
 */
export module Compute.OperationTraits.Template;

export import Compute.DeviceType;
export import Compute.OperationType;
export import Dnn.TensorDataType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Primary traits template for unified compile-time operation dispatch.
     *
     * Each specialization must provide a nested `type` alias naming the concrete
     * operation class for the given (OperationType, DeviceType, precision, policy)
     * combination. The primary template is intentionally undefined -- an unspecialized
     * instantiation is a hard compile error.
     *
     * @tparam TOp          Operation identifier from the OperationType enum.
     * @tparam TDeviceType  Target device (Cpu, Cuda, ...).
     * @tparam TPrecision   Compute and activation precision.
     * @tparam TPolicy      Optional policy type. Defaults to void for policy-free ops.
     */
    export template<OperationType TOp, DeviceType TDeviceType, TensorDataType TPrecision, typename TPolicy = void>
    struct OperationTraits;

    // -------------------------------------------------------------------------
    // Per-operation signature concepts.
    //
    // Each concept enforces the concrete method signatures required by the
    // corresponding component. Satisfied by the op type before the component
    // stores it in its local OpType alias.
    // -------------------------------------------------------------------------

    /**
     * @brief Contract for LinearOp: typed forward matmul and backward weight/input gradients.
     *
     * @tparam TOp     Candidate op type.
     * @tparam TTensor Tensor type at the call site.
     */
    export template<typename TOp, typename TTensor>
    concept LinearOpConcept = requires( const TOp& op, const TTensor& in, TTensor& out )
    {
        op.forward( in, out );
        op.backward( in, in, out );
    };

    /**
     * @brief Contract for GroupedQueryAttentionOp: positional forward and backward.
     *
     * @tparam TOp     Candidate op type.
     * @tparam TTensor Tensor type at the call site.
     */
    export template<typename TOp, typename TTensor>
    concept GqaOpConcept = requires( const TOp& op, const TTensor& in, TTensor& out )
    {
        op.forward( in, out );
        op.backward( in, in, out );
    };

    /**
     * @brief Contract for policy-free unary ops (Softmax, RmsNorm, LayerNorm, Residual, ...).
     *
     * @tparam TOp     Candidate op type.
     * @tparam TTensor Tensor type at the call site.
     */
    export template<typename TOp, typename TTensor>
    concept UnaryOpConcept = requires( const TOp& op, const TTensor& in, TTensor& out )
    {
        op.forward( in, out );
    };

    /**
     * @brief Contract for SamplingOp: in-place token sampling from a logits tensor.
     *
     * temperature and top_k are per-call parameters -- no separate configure() step.
     * token_out is a device INT32 tensor written in-place; the caller provides the
     * buffer (typically decode_token_device_ in LlamaModel).
     *
     * Non-const because CpuSamplingOp holds an mt19937 rng_ updated on each call.
     *
     * @tparam TOp      Candidate op type.
     * @tparam TLogits  Logits tensor type (model compute precision).
     * @tparam TToken   Output tensor type (INT32 device tensor).
     */
    export template<typename TOp, typename TLogits, typename TToken>
    concept SamplingOpConcept = requires( TOp& op,
                                          const TLogits& logits, TToken& token_out,
                                          float temperature, int top_k )
    {
        op.forward( logits, token_out, temperature, top_k );
    };

}  // namespace Mila::Dnn::Compute
