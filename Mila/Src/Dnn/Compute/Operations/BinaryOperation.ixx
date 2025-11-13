/**
 * @file BinaryOperation.ixx
 * @brief Abstract device-agnostic binary operation interface.
 *
 * Typed, device-aware interface for binary operations (two inputs -> one output).
 */

module;
#include <type_traits>
#include <utility>

export module Compute.BinaryOperation;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.CudaDeviceMemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
    /**
     * @tparam TDeviceType Device type for this operation (Cpu, Cuda, ...)
     * @tparam TPrecision Canonical element precision produced/consumed by the op (e.g. FP32)
     * @tparam TInputA Element type for left input (defaults to TPrecision)
     * @tparam TInputB Element type for right input (defaults to TPrecision)
     *
     * @note Operation precision (TPrecision) is constrained to be supported on the target device.
     */
    export template <
        DeviceType TDeviceType,
        TensorDataType TPrecision,
        TensorDataType TInputA = TPrecision,
        TensorDataType TInputB = TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class BinaryOperation : public Operation<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

        // Concrete tensor aliases for implementers
        using TensorOutputType   = Tensor<TPrecision, MR>;
        using TensorLeftType     = Tensor<TInputA, MR>;
        using TensorRightType    = Tensor<TInputB, MR>;

        // Parameter / gradient aliases (parameters normally use op precision)
        using ParameterTensor    = Tensor<TPrecision, MR>;
        using ParameterGradTensor= Tensor<TPrecision, MR>;

        virtual ~BinaryOperation() = default;

        /**
         * @brief Forward pass: output = f(input_a, input_b)
         *
         * Implementations should accept polymorphic `ITensor` references and
         * may use the typed helpers below to obtain concrete `Tensor<T,...>` references.
         */
        virtual void forward( const ITensor& input_a, const ITensor& input_b, ITensor& output ) const = 0;

        /**
         * @brief Backward pass: compute gradients w.r.t. inputs.
         *
         * Parameter gradients are written via pointers bound by setParameterGradients().
         *
         * @param input_a First input from forward pass
         * @param input_b Second input from forward pass
         * @param output_grad Gradient w.r.t. output (dL/dOutput)
         * @param input_a_grad Gradient w.r.t. first input (dL/dInputA)
         * @param input_b_grad Gradient w.r.t. second input (dL/dInputB)
         */
        virtual void backward(
            const ITensor& input_a,
            const ITensor& input_b,
            const ITensor& output_grad,
            ITensor& input_a_grad,
            [[maybe_unused]] ITensor& input_b_grad ) const = 0;

    protected:
        // Typed dynamic-cast helpers to avoid manual rawData() casts.
        static const TensorLeftType&  asLeftTensor( const ITensor& t )
        {
            return dynamic_cast<const TensorLeftType&>(t);
        }

        static const TensorRightType& asRightTensor( const ITensor& t )
        {
            return dynamic_cast<const TensorRightType&>(t);
        }

        static TensorOutputType& asOutputTensor( ITensor& t )
        {
            return dynamic_cast<TensorOutputType&>(t);
        }
    };
}