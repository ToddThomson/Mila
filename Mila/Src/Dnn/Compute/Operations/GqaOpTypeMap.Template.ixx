/**
 * @file GqaOpDispatch.Template.ixx
 * @brief Compile-time dispatch mapping (DeviceType, TComputePrecision, TWeight) to the 
 * concrete GqaOp compaute backend type.
 *
 * Backend specializations live co-located with their concrete op and import this primary template.
 */
//export module Compute.GqaOpTypeMap.Template;
//
//import Compute.DeviceType;
//import Dnn.TensorDataType;
//import Dnn.Tensor;
//import Dnn.ITensor;
//
//namespace Mila::Dnn::Compute
//{
//    /**
//     * @brief Primary traits template for GQA operation dispatch.
//     *
//     * Specializations must provide a nested `type` alias naming the concrete op class.
//     * An unspecialized instantiation is a hard compile error — the correct diagnostic
//     * for an unsupported (DeviceType, TComputePrecision, TKvCache) combination.
//     *
//     * @tparam TDeviceType  Target device.
//     * @tparam TComputePrecision   Activation and compute precision.
//     * @tparam TKvCache      KV Cache storage type. Defaults to TComputePrecision for non-quantized paths.
//     */
//    export template<DeviceType TDeviceType, TensorDataType TComputePrecision, TensorDataType TKvCache = TComputePrecision>
//    class GqaOpTypeMap;
//
//    /**
//     * @brief Concept enforcing the forward/backward signature contract for Linear ops.
//     *
//     * Satisfied by any type that exposes forward(in, out) and backward(in, grad, in_grad).
//     * Works with both ITensor-based and typed-tensor signatures since TTensor derives from ITensor.
//     */
//    export template<typename TOp, typename TTensor>
//        concept GqaOpConcept = requires(const TOp & op, const TTensor & in, TTensor & out)
//    {
//        op.forward( in, out );
//        op.backward( in, in, out );
//    };
//}