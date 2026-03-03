/**
 * @file PairedOperation.ixx
 * @brief Abstract device-agnostic paired operation interface.
 *
 * Typed, device-aware interface for paired operations (two inputs -> two outputs).
 * Models symmetric transforms such as RoPE and its variants.
 */

module;
#include <type_traits>

export module Compute.PairedOperation;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.Precision;
import Compute.Device;
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
     * @brief Abstract base for paired operations: two inputs -> two outputs.
     *
     * Models symmetric transforms where two tensors are processed jointly
     * and each produces an independent output (e.g. Q and K in RoPE).
     *
     * The backward signature is symmetric with forward: 2-in / 2-out. Implementations
     * that require saved forward activations for their backward pass must cache them
     * internally during forward().
     *
     * @tparam TDeviceType Device target (DeviceType::Cpu, DeviceType::Cuda, ...)
     * @tparam TPrecision  Canonical element precision for inputs and outputs (e.g. FP32)
     * @tparam TInputA     Element type for the first input tensor (defaults to TPrecision)
     * @tparam TInputB     Element type for the second input tensor (defaults to TInputA)
     */
    export template <
        DeviceType TDeviceType,
        TensorDataType TPrecision,
        TensorDataType TInputA = TPrecision,
        TensorDataType TInputB = TInputA>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class PairedOperation : public Operation<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

        using TensorOutputType = Tensor<TPrecision, MR>;
        using TensorInputAType = Tensor<TInputA, MR>;
        using TensorInputBType = Tensor<TInputB, MR>;

        virtual ~PairedOperation() = default;

        /**
         * @brief Forward pass: (out_a, out_b) = f(in_a, in_b)
         *
         * @param in_a  First input tensor.
         * @param in_b  Second input tensor.
         * @param out_a First output tensor (same shape as in_a).
         * @param out_b Second output tensor (same shape as in_b).
         */
        virtual void forward(
            const ITensor& in_a, const ITensor& in_b,
            ITensor& out_a, ITensor& out_b ) const = 0;

        /**
         * @brief Backward pass: propagate upstream gradients to input gradients.
         *
         * @param grad_out_a Upstream gradient w.r.t. out_a (dL/dout_a).
         * @param grad_out_b Upstream gradient w.r.t. out_b (dL/dout_b).
         * @param grad_in_a  Output gradient w.r.t. in_a  (dL/din_a).
         * @param grad_in_b  Output gradient w.r.t. in_b  (dL/din_b).
         */
        virtual void backward(
            const ITensor& grad_out_a, const ITensor& grad_out_b,
            ITensor& grad_in_a, ITensor& grad_in_b ) const = 0;

    protected:
        static const TensorInputAType& asInputA( const ITensor& t )
        {
            return dynamic_cast<const TensorInputAType&>(t);
        }

        static const TensorInputBType& asInputB( const ITensor& t )
        {
            return dynamic_cast<const TensorInputBType&>(t);
        }

        static TensorOutputType& asOutputTensor( ITensor& t )
        {
            return dynamic_cast<TensorOutputType&>(t);
        }
    };
}