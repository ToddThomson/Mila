/**
 * @file LinearOpDispatch.Template.ixx
 * @brief Compile-time dispatch mapping (DeviceType, TPrecision, TWeight) to the concrete LinearOp type.
 *
 * Backend specializations live co-located with their concrete op and import this primary template.
 */
export module Compute.LinearOpTypeMap.Template;

import Compute.DeviceType;
import Dnn.TensorDataType;
import Dnn.Tensor;
import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Primary traits template for Linear operation dispatch.
     *
     * Specializations must provide a nested `type` alias naming the concrete op class.
     * An unspecialized instantiation is a hard compile error — the correct diagnostic
     * for an unsupported (DeviceType, TPrecision, TWeight) combination.
     *
     * @tparam TDeviceType  Target device.
     * @tparam TPrecision   Activation and compute precision.
     * @tparam TWeight      Weight storage type. Defaults to TPrecision for non-quantized paths.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, TensorDataType TWeight = TPrecision>
    class LinearOpTypeMap;

    /**
     * @brief Concept enforcing the forward/backward signature contract for Linear ops.
     *
     * Satisfied by any type that exposes forward(in, out) and backward(in, grad, in_grad).
     * Works with both ITensor-based and typed-tensor signatures since TTensor derives from ITensor.
     */
    export template<typename TOp, typename TTensor>
        concept LinearOpConcept = requires(const TOp & op, const TTensor & in, TTensor & out)
    {
        op.forward( in, out );
        op.backward( in, in, out );
    };
}