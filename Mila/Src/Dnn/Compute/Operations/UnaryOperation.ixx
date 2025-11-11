/**
 * @file UnaryOperation.ixx
 * @brief Device-agnostic unary operation interface using abstract tensor data types
 *
 * Provides a typed, device-aware interface for unary ops. Supports a distinct
 * input element type (`TInput`) which may differ from the operation precision
 * (useful for index/mask inputs).
 */

module;
#include <type_traits>

export module Compute.UnaryOperation;

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
import Compute.DeviceTypeTraits.Cpu;
import Compute.CudaDeviceMemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.OperationAttributes;

namespace Mila::Dnn::Compute
{
    /**
     * @tparam TDeviceType Device target for the operation (DeviceType::Cpu, DeviceType::Cuda, ...)
     * @tparam TPrecision Canonical element precision produced/consumed by the op (e.g. FP32)
     * @tparam TInput Optional element type for the runtime input tensor (defaults to TPrecision,
     *                e.g. INT32 for token indices while TPrecision == FP32)
     *
     * @note Constrains are applied to ensure the operation precision is supported on the device.
     */
    export template <DeviceType TDeviceType, TensorDataType TInput, TensorDataType TPrecision = TInput>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class UnaryOperation : public Operation<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

        // Concrete tensor aliases for implementers to use (typed, device-aware)
        using TensorOutputType = Tensor<TPrecision, MR>;
        using TensorInputType  = Tensor<TInput, MR>;

        virtual ~UnaryOperation() = default;

        /**
         * @brief Forward pass: compute output = f(input)
         *
         * Implementations should accept polymorphic ITensor references and may
         * use the typed aliases / helpers to obtain typed tensor references.
         */
        virtual void forward( const ITensor& input, ITensor& output ) const = 0;

        /**
         * @brief Backward pass: compute gradient wrt input given output gradient.
         *
         * Signature ordered as (input, output_grad, input_grad) to match module
         * and operation implementations across the codebase.
         */
        virtual void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) const = 0;

    protected:
        // Helpers for typed dynamic casts to concrete Tensor<T,...> types.
        // Use these to avoid unsafe void* casts and to prefer the typed `.data()` accessor.
        static const TensorInputType& asInputTensor( const ITensor& t )
        {
            return dynamic_cast<const TensorInputType&>(t);
        }

        static TensorOutputType& asOutputTensor( ITensor& t )
        {
            return dynamic_cast<TensorOutputType&>(t);
        }
    };
}