/**
 * @file BinaryOperation.ixx
 * @brief Abstract device-agnostic binary operation interface.
 *
 * Provides the typed abstract interface used by device-specific binary operation
 * implementations. See member documentation for API contracts and semantics.
 */

module;
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

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
import Compute.OperationAttributes;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract interface for binary operations (two inputs ? one output).
     *
     * Inherits parameter management from OperationBase:
     * - setParameters() binds weight/bias pointers before execution
     * - setParameterGradients() binds gradient pointers before execution
     * - build() performs shape-dependent setup and caches dimension info
     *
     * State management:
     * - Operations maintain internal state (cached probabilities, masks, etc.)
     * - State NEVER appears in forward/backward signatures
     * - Implementations cache state as private members between forward/backward
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template <DeviceType TDeviceType, TensorDataType TPrecision>
        class BinaryOperation : public OperationBase<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using TensorType = Tensor<TPrecision, MR>;

        virtual ~BinaryOperation() = default;

        /**
         * @brief Forward pass: output = f(inputA, inputB).
         *
         * Parameters are accessed via pointers bound by setParameters().
         * Intermediate state is cached internally by the operation.
         *
         * @param inputA First input tensor
         * @param inputB Second input tensor
         * @param output Output tensor (allocated/resized by implementation)
         */
        virtual void forward( const ITensor& input_a, const ITensor& input_b, ITensor& output ) const = 0;

        /**
         * @brief Backward pass: compute gradients w.r.t. inputs.
         *
         * Parameter gradients are written via pointers bound by setParameterGradients().
         * Intermediate state is accessed from operation's internal cache.
         *
         * @param inputA First input from forward pass
         * @param inputB Second input from forward pass
         * @param output_gradient Gradient w.r.t. output (dL/dOutput)
         * @param inputA_gradient Gradient w.r.t. first input (dL/dInputA)
         * @param inputB_gradient Gradient w.r.t. second input (dL/dInputB)
         *
         * Notes:
         * - Some operations may not compute gradients for all inputs
         *   (e.g., CrossEntropy doesn't differentiate w.r.t. integer targets)
         * - Implementations should document accumulation semantics
         * - Parameter gradients are accumulated via cached pointers
         */
        virtual void backward(
            const ITensor& inputA,
            const ITensor& inputB,
            const ITensor& output_gradient,
            ITensor& inputA_gradient,
            ITensor& inputB_gradient ) const = 0;
    };
}