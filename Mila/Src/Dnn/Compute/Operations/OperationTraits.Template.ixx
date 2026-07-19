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
     * Each specialization provides either a nested `type` alias naming the concrete
     * operation class, or (for functor-templated ops such as ElementwiseActivationOp)
     * a nested `op_for<Functor>` alias, for the given (OperationType, DeviceType,
     * precision, policy) combination.
     *
     * The primary template is intentionally left **undefined**. An unsupported tuple
     * therefore names an incomplete type, and the compiler reports a single-line
     * "use of undefined type OperationTraits<Op, Device, Precision, Policy>" naming the
     * exact tuple -- a readable diagnostic, not a multi-hundred-line constraint cascade.
     * (A kernel that only supports FP32 must not advertise a BF16 row: the honest
     * failure is a missing specialization here, not a poisoned row whose op fails a
     * `float || half` kernel constraint deep in dispatch.)
     *
     * To branch on availability at compile time -- e.g. a multi-precision typed test
     * skipping the precisions an op does not implement -- use the SFINAE-safe
     * `OperationSupported<...>` predicate below rather than instantiating this template.
     * The primary must stay undefined for that predicate to work: a `static_assert` in
     * the primary body would fire during the predicate's own probe, turning the
     * detectable "false" back into a hard error.
     *
     * @tparam TOp          Operation identifier from the OperationType enum.
     * @tparam TDeviceType  Target device (Cpu, Cuda, ...).
     * @tparam TPrecision   Compute and activation precision.
     * @tparam TPolicy      Optional policy type. Defaults to void for policy-free ops.
     */
    export template<OperationType TOp, DeviceType TDeviceType, TensorDataType TPrecision, typename TPolicy = void>
    struct OperationTraits;

    /**
     * @brief True iff a concrete OperationTraits specialization exists for the tuple.
     *
     * SFINAE-safe: probes whether OperationTraits<...> is a complete type (a matching
     * specialization completes it; the undefined primary stays incomplete for every
     * unsupported tuple). Satisfied for both `type`-bearing and `op_for`-bearing
     * specializations. Usable in `if constexpr` and `static_assert` to skip or reject
     * (Op, Device, Precision, Policy) combinations without triggering a hard error --
     * the seam a multi-precision typed test uses to sweep only the precisions an op
     * actually implements.
     *
     * @tparam TOp          Operation identifier from the OperationType enum.
     * @tparam TDeviceType  Target device (Cpu, Cuda, ...).
     * @tparam TPrecision   Compute and activation precision.
     * @tparam TPolicy      Optional policy type. Defaults to void for policy-free ops.
     */
    export template<OperationType TOp, DeviceType TDeviceType, TensorDataType TPrecision, typename TPolicy = void>
    concept OperationSupported = requires { sizeof( OperationTraits<TOp, TDeviceType, TPrecision, TPolicy> ); };

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
     * Per-call sampling knobs travel in a SamplingParams struct (temperature/top_k/
     * top_p/seed) -- a struct, not loose scalars, so adding a filter does not churn
     * the signature. The host-drawn uniform `r` is passed as a scalar, keeping the op
     * pure and deterministic. token_out is a device INT32 tensor written in-place; the
     * caller provides the buffer (the model's decode_token_device_).
     *
     * @tparam TOp      Candidate op type.
     * @tparam TLogits  Logits tensor type (model compute precision).
     * @tparam TToken   Output tensor type (INT32 device tensor).
     * @tparam TParams  Per-call sampling parameter struct (SamplingParams).
     */
    export template<typename TOp, typename TLogits, typename TToken, typename TParams>
    concept SamplingOpConcept = requires( const TOp& op,
                                          const TLogits& logits, TToken& token_out,
                                          const TParams& params, float r )
    {
        op.forward( logits, token_out, params, r );
    };

}  // namespace Mila::Dnn::Compute
