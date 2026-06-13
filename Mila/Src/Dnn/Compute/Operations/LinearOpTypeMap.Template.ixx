// RETIRED (Alpha.6): superseded by compile-time OperationTraits dispatch (Compute.OperationTraits);
// removed from the build, retained on disk for reference. See ROADMAP/BACKLOG Alpha.6 dispatch close-out.

/**
 * @file LinearOpTypeMap.Template.ixx
 * @brief Primary compile-time dispatch template mapping (DeviceType, TPrecision, TWeightQuant)
 *        to a concrete LinearOp type.
 *
 * Backend specializations live in LinearOpTypeMap.Cpu.ixx and LinearOpTypeMap.Cuda.ixx and
 * import this primary template. A missing specialization is a hard compile error.
 */
//export module Compute.LinearOpTypeMap.Template;
//
//import Compute.DeviceType;
//import Dnn.TensorDataType;
//import Dnn.Quantization.Weight.Policies;
//
//namespace Mila::Dnn::Compute
//{
//    using namespace Mila::Dnn::Quant::Weight;
//
//    /**
//     * @brief Primary traits template for compile-time Linear operation dispatch.
//     *
//     * Each specialization must provide a nested `op_type` alias naming the concrete op class
//     * for the given (DeviceType, compute precision, weight quantization policy) combination.
//     * An unspecialized instantiation is a hard compile error � the correct diagnostic for an
//     * unsupported combination.
//     *
//     * @tparam TDeviceType   Target device.
//     * @tparam TPrecision    Activation and accumulation precision.
//     * @tparam TWeightQuant  Weight quantization policy. Defaults to NoWeightQuant for all
//     *                       standard non-quantized paths.
//     */
//    export template<DeviceType TDeviceType, TensorDataType TPrecision, WeightQuantPolicy TWeightQuant = NoWeightQuant>
//    class LinearOpTypeMap;
//
//    /**
//     * @brief Concept enforcing the forward/backward signature contract for Linear ops.
//     *
//     * Satisfied by any type exposing forward(in, out) and backward(in, grad, in_grad).
//     * Compatible with both ITensor-based and typed-tensor signatures.
//     *
//     * @tparam TOp     Candidate op type.
//     * @tparam TTensor Tensor type used at the call site.
//     */
//    export template<typename TOp, typename TTensor>
//    concept LinearOpConcept = requires(const TOp& op, const TTensor& in, TTensor& out)
//    {
//        op.forward( in, out );
//        op.backward( in, in, out );
//    };
//}